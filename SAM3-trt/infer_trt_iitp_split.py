import argparse
import collections
import importlib.util
import json
import os
import time

import cv2
import numpy as np
import PIL.Image
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from tokenizers import Tokenizer


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT_ENGINE_ROOT = os.environ.get(
    "SAM3_SPLIT_ENGINE_ROOT",
    os.path.join(ROOT, "weights", "SAM3-trt", "split_engines"),
)


_spec = importlib.util.spec_from_file_location(
    "iitp_common",
    os.path.join(ROOT, "SAM3-trt", "infer_trt_iitp.py"),
)
iitp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iitp)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

PROMPT_SPECS = (
    ("transparent", "plastic beverage bottle"),
    ("metal", "recyclable metal packaging"),
    ("metal", "crushed can"),
    ("Cardboard", "cardboard box"),
    ("Cardboard", "cardboard package"),
)


def trt_dtype_to_np(dtype):
    return np.dtype(trt.nptype(dtype))


class SplitEngine:
    def __init__(self, path):
        self.path = path
        self.stream = cuda.Stream()
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.is_input = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.names}
        self.dtype = {n: trt_dtype_to_np(self.engine.get_tensor_dtype(n)) for n in self.names}
        self.shape = {n: tuple(int(x) for x in self.engine.get_tensor_shape(n)) for n in self.names}
        self.loc = {n: self.engine.get_tensor_location(n) for n in self.names}
        self.dptr = {}
        self.hbuf = {}

    def allocate(self, host_outputs=()):
        host_outputs = set(host_outputs)
        for name in self.names:
            shape = self.shape[name]
            dtype = self.dtype[name]
            nbytes = int(np.prod(shape)) * dtype.itemsize
            if self.loc[name] == trt.TensorLocation.HOST:
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)
                self.hbuf[name] = host
                self.context.set_tensor_address(name, int(host.ctypes.data))
            else:
                dev = cuda.mem_alloc(nbytes)
                self.dptr[name] = dev
                self.context.set_tensor_address(name, int(dev))
                if (not self.is_input[name]) and name in host_outputs:
                    self.hbuf[name] = cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)

    def set_device_input(self, name, src_engine, src_name):
        self.context.set_tensor_address(name, int(src_engine.dptr[src_name]))

    def enqueue(self, inputs, stream):
        for name, arr in inputs.items():
            if arr.dtype != self.dtype[name]:
                arr = arr.astype(self.dtype[name], copy=False)
            if tuple(arr.shape) != self.shape[name]:
                arr = arr.reshape(self.shape[name])
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            if self.loc[name] == trt.TensorLocation.HOST:
                np.copyto(self.hbuf[name], arr)
            else:
                cuda.memcpy_htod_async(self.dptr[name], arr, stream)
        ok = self.context.execute_async_v3(stream.handle)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed: {self.path}")

    def copy_output(self, name, stream):
        cuda.memcpy_dtoh_async(self.hbuf[name], self.dptr[name], stream)


class SplitSam3Runner:
    def __init__(self, args):
        self.encoder = SplitEngine(args.image_encoder)
        self.decoder = SplitEngine(args.decoder)
        self.languages = [SplitEngine(args.language_encoder) for _ in PROMPT_SPECS]
        self.tokenizer = Tokenizer.from_file(args.tokenizer)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)
        self.encoder.allocate()
        self.decoder.allocate(host_outputs=("scores", "boxes", "masks_logits"))
        for lang in self.languages:
            lang.allocate()
        self.stream = cuda.Stream()
        self._prime_language()

    def _prime_language(self):
        for lang, (_cat, prompt) in zip(self.languages, PROMPT_SPECS):
            encoded = self.tokenizer.encode(prompt)
            tokens = np.asarray([encoded.ids], dtype=np.int64)
            lang.enqueue({"tokens": tokens}, self.stream)
        self.stream.synchronize()

    @staticmethod
    def preprocess_image(image):
        resized = image.resize((1008, 1008), resample=PIL.Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
        return np.ascontiguousarray(arr)

    def _bind_decoder_common(self, lang):
        self.decoder.set_device_input("vision_pos_enc_2", self.encoder, "vision_pos_enc_2")
        self.decoder.set_device_input("backbone_fpn_0", self.encoder, "backbone_fpn_0")
        self.decoder.set_device_input("backbone_fpn_1", self.encoder, "backbone_fpn_1")
        self.decoder.set_device_input("backbone_fpn_2", self.encoder, "backbone_fpn_2")
        self.decoder.set_device_input("language_mask", lang, "text_attention_mask")
        self.decoder.set_device_input("language_features", lang, "text_memory")

    def _copy_selected_masks(self, keep_indices, stream):
        out = self.decoder.hbuf["masks_logits"]
        mask_h, mask_w = out.shape[-2], out.shape[-1]
        itemsize = out.dtype.itemsize
        mask_nbytes = mask_h * mask_w * itemsize
        base = int(self.decoder.dptr["masks_logits"])
        for idx in keep_indices:
            cuda.memcpy_dtoh_async(out[int(idx)], base + int(idx) * mask_nbytes, stream)

    def infer_image(self, image, score_thr=0.5, profile=False):
        x = self.preprocess_image(image)
        prompt_results = {
            "transparent": iitp.empty_prompt_result(image.height, image.width),
            "metal": iitp.empty_prompt_result(image.height, image.width),
            "Cardboard": iitp.empty_prompt_result(image.height, image.width),
        }

        box_coords = np.zeros((1, 1, 4), dtype=np.float32)
        box_labels = np.ones((1, 1), dtype=np.int32)
        box_masks = np.ones((1, 1), dtype=np.int32)

        e0 = cuda.Event()
        e1 = cuda.Event()
        e2 = cuda.Event()
        e3 = cuda.Event()
        e0.record(self.stream)
        self.encoder.enqueue({"image": x}, self.stream)
        e1.record(self.stream)

        copied_masks = 0
        for lang, (category, prompt) in zip(self.languages, PROMPT_SPECS):
            self._bind_decoder_common(lang)
            self.decoder.enqueue(
                {
                    "box_coords": box_coords,
                    "box_labels": box_labels,
                    "box_masks": box_masks,
                },
                self.stream,
            )
            self.decoder.copy_output("scores", self.stream)
            self.decoder.copy_output("boxes", self.stream)
            self.stream.synchronize()

            scores_all = self.decoder.hbuf["scores"].copy()
            keep = np.flatnonzero(scores_all > score_thr)
            if keep.size > 0:
                self._copy_selected_masks(keep, self.stream)
                self.stream.synchronize()
                copied_masks += int(keep.size)
                masks_logits = self.decoder.hbuf["masks_logits"][keep].copy()
                scores = scores_all[keep].astype(np.float32)
                masks_prob = np.empty((keep.size, image.height, image.width), dtype=np.float32)
                for i, mask_logit in enumerate(masks_logits):
                    resized = cv2.resize(mask_logit.astype(np.float32), (image.width, image.height), interpolation=cv2.INTER_LINEAR)
                    masks_prob[i] = iitp.sig(resized).astype(np.float32)
                processed_masks, processed_indices = iitp.process_masks_by_size(
                    masks_prob >= 0.5,
                    threshold=1000,
                    prefer="smaller",
                )
                masks_prob = processed_masks.astype(np.float32)
                scores = scores[processed_indices] if processed_indices.size > 0 else scores[:0]
            else:
                masks_prob = np.empty((0, image.height, image.width), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)

            iitp.append_prompt_result(
                prompt_results[category],
                {
                    "masks": masks_prob,
                    "scores": scores,
                    "raw_count": int(keep.size),
                    "mask_nms_removed": 0,
                    "prompts": [prompt],
                },
            )

        e2.record(self.stream)
        e2.synchronize()
        if profile:
            prof = {
                "encoder": float(e0.time_till(e1)),
                "decoder_total": float(e1.time_till(e2)),
                "total_cuda": float(e0.time_till(e2)),
                "copied_masks": copied_masks,
            }
        else:
            prof = {}
        return prompt_results, prof


def collect_images(image_dir, limit=None):
    paths = iitp.collect_image_paths(image_dir, limit=limit)
    if not paths:
        raise FileNotFoundError(f"No images found: {image_dir}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="IITP SAM3 split TensorRT inference.")
    parser.add_argument("--image-dir", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "images"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "output", "overlay_outputs_iitp_split"))
    parser.add_argument("--image-encoder", default=os.path.join(SPLIT_ENGINE_ROOT, "sam3_image_encoder_bf16.plan"))
    parser.add_argument("--language-encoder", default=os.path.join(SPLIT_ENGINE_ROOT, "sam3_language_encoder_bf16.plan"))
    parser.add_argument("--decoder", default=os.path.join(SPLIT_ENGINE_ROOT, "sam3_decoder_bf16.plan"))
    parser.add_argument("--tokenizer", default=os.path.join(ROOT, "weights", "usls", "tokenizer.json"))
    parser.add_argument("--annotations", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "annotations_crop.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--transparent-min-score", type=float, default=iitp.DEFAULT_CATEGORY_MIN_SCORES["transparent"])
    parser.add_argument("--metal-min-score", type=float, default=iitp.DEFAULT_CATEGORY_MIN_SCORES["metal"])
    parser.add_argument("--cardboard-min-score", type=float, default=iitp.DEFAULT_CATEGORY_MIN_SCORES["Cardboard"])
    parser.add_argument("--merge-nms-iou", type=float, default=0.5)
    parser.add_argument("--no-save-overlays", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = collect_images(args.image_dir, args.limit)
    runner = SplitSam3Runner(args)
    pred_rows = []
    timings = []
    pred_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    min_scores = {
        "transparent": args.transparent_min_score,
        "metal": args.metal_min_score,
        "Cardboard": args.cardboard_min_score,
    }
    with open(pred_jsonl, "w", encoding="utf-8") as f:
        for idx, path in enumerate(paths, start=1):
            t0 = time.perf_counter()
            image = PIL.Image.open(path).convert("RGB")
            t1 = time.perf_counter()
            prompt_results, prof = runner.infer_image(image, score_thr=args.score_thr, profile=args.profile)
            t2 = time.perf_counter()
            row = iitp.finalize_prompt_results_row(
                image,
                path,
                prompt_results,
                out_dir=args.out_dir,
                category_min_scores=min_scores,
                merge_nms_iou=args.merge_nms_iou,
                mask_thr=0.5,
                top_k=None,
                suppress_conveyor=False,
                mask_nms=True,
                mask_nms_iou=0.3,
                mask_nms_overlap=0.6,
                save_overlay=not args.no_save_overlays,
                quiet=True,
            )
            t3 = time.perf_counter()
            timing = {
                "preprocess": (t1 - t0) * 1000.0,
                "inference": (t2 - t1) * 1000.0,
                "postprocess_save": (t3 - t2) * 1000.0,
                "total": (t3 - t0) * 1000.0,
            }
            row["timing_ms"] = timing
            if args.profile:
                row["split_profile_ms"] = prof
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_rows.append(row)
            timings.append(timing)
            if not args.quiet:
                print(f"[{idx}/{len(paths)}] {os.path.basename(path)} total={timing['total']:.2f} infer={timing['inference']:.2f}")

    print(f"saved predictions: {pred_jsonl}")
    if timings:
        avg = {k: sum(t[k] for t in timings) / len(timings) for k in timings[0].keys()}
        print(
            "avg_time_ms "
            f"preprocess={avg['preprocess']:.2f} "
            f"infer={avg['inference']:.2f} "
            f"post={avg['postprocess_save']:.2f} "
            f"total={avg['total']:.2f}"
        )
    if not args.no_eval:
        gt = iitp.load_annotations_jsonl(args.annotations)
        metrics = iitp.evaluate_predictions(gt, pred_rows, iou_thr=0.5, y_min=40.0, y_max=440.0, y_filter_mode="center")
        metrics_path = os.path.join(args.out_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        iitp.print_eval_metrics(metrics)
        print(f"saved eval metrics: {metrics_path}")


if __name__ == "__main__":
    main()
