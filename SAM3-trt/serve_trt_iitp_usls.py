import argparse
import importlib.util
import json
import os
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
import PIL.Image
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from tokenizers import Tokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
IITP_COMMON_PATH = os.path.join(SCRIPT_DIR, "infer_trt_iitp.py")

_spec = importlib.util.spec_from_file_location("iitp_common", IITP_COMMON_PATH)
iitp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iitp)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CATEGORY_TO_ID = {
    "transparent": 1,
    "metal": 2,
    "Cardboard": 3,
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}

DEFAULT_PROMPT_SPECS = (
    ("transparent", "transparent plastic bottle"),
    ("metal", "beverage can"),
    ("metal", "aluminum foil wrapper"),
    ("metal", "foil packaging"),
    ("Cardboard", "cardboard box"),
    ("Cardboard", "cardboard package"),
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def trt_dtype_to_np(dtype):
    return np.dtype(trt.nptype(dtype))


def recvall(sock, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def send_packet(sock, meta, payload=b""):
    meta_b = json.dumps(meta).encode("utf-8")
    sock.sendall(struct.pack("!IQ", len(meta_b), len(payload)))
    sock.sendall(meta_b)
    if payload:
        sock.sendall(payload)


def recv_packet(sock):
    meta_len, payload_len = struct.unpack("!IQ", recvall(sock, 12))
    meta = json.loads(recvall(sock, meta_len).decode("utf-8"))
    payload = recvall(sock, payload_len) if payload_len else b""
    return meta, payload


def pack_arrays(arrays, msg_type, extra_meta=None):
    meta = {"type": msg_type, "arrays": {}, "order": "C"}
    if extra_meta:
        meta.update(extra_meta)
    chunks = []
    offset = 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        nbytes = int(arr.nbytes)
        meta["arrays"][name] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "offset": offset,
            "nbytes": nbytes,
        }
        chunks.append(arr.tobytes(order="C"))
        offset += nbytes
    return meta, b"".join(chunks)


def unpack_arrays(meta, payload):
    arrays = {}
    for name, info in meta.get("arrays", {}).items():
        offset = int(info["offset"])
        nbytes = int(info["nbytes"])
        dtype = np.dtype(info["dtype"])
        shape = tuple(info["shape"])
        arrays[name] = np.frombuffer(payload[offset:offset + nbytes], dtype=dtype).reshape(shape).copy()
    return arrays


def decode_image_packet(meta, payload):
    msg_type = meta.get("type", "frame")
    filename = meta.get("filename", f"{int(meta.get('idx', 0)):06d}.jpg")
    if "arrays" in meta:
        arrays = unpack_arrays(meta, payload)
        if "rgb" in arrays:
            image_np = arrays["rgb"]
            color_order = "rgb"
        elif "image" in arrays:
            image_np = arrays["image"]
            color_order = str(meta.get("color_order", "rgb")).lower()
        elif "bgr" in arrays:
            image_np = arrays["bgr"]
            color_order = "bgr"
        else:
            raise ValueError("array packet must contain one of: rgb, bgr, image")
        image_np = np.asarray(image_np)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"image array must be HxWx3, got {image_np.shape}")
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        if color_order == "bgr":
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(image_np), filename

    if msg_type in {"jpeg", "jpg", "image_jpeg", "png", "image_png", "frame_jpeg"}:
        arr = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("failed to decode compressed image payload")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(rgb), filename

    raise ValueError(f"unsupported packet type: {msg_type}")


class TrtEngine:
    def __init__(self, path, input_shapes=None, host_outputs=()):
        self.path = path
        self.stream = cuda.Stream()
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to load TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create TensorRT context: {path}")

        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.is_input = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.names}
        self.dtype = {n: trt_dtype_to_np(self.engine.get_tensor_dtype(n)) for n in self.names}
        self.loc = {n: self.engine.get_tensor_location(n) for n in self.names}
        self.dptr = {}
        self.hbuf = {}
        self.shape = {}

        input_shapes = input_shapes or {}
        for name, shape in input_shapes.items():
            self.context.set_input_shape(name, tuple(shape))
        for name in self.names:
            if self.is_input[name] and name not in input_shapes:
                shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
                if any(x < 0 for x in shape):
                    raise ValueError(f"dynamic input shape for {name} must be provided: {path}")
                self.context.set_input_shape(name, shape)

        self.allocate(host_outputs=host_outputs)

    def tensor_shape(self, name):
        shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
        if any(x < 0 for x in shape):
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
        return shape

    def allocate(self, host_outputs=()):
        host_outputs = set(host_outputs)
        for name in self.names:
            shape = self.tensor_shape(name)
            dtype = self.dtype[name]
            self.shape[name] = shape
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

    def bind_from(self, dst_name, src_engine, src_name, byte_offset=0):
        self.context.set_tensor_address(dst_name, int(src_engine.dptr[src_name]) + int(byte_offset))

    def copy_output(self, name, stream):
        cuda.memcpy_dtoh_async(self.hbuf[name], self.dptr[name], stream)


class UslsSam3Runner:
    def __init__(self, args):
        self.prompt_specs = args.prompt_specs
        self.prompt_categories = tuple(dict.fromkeys(category for category, _prompt in self.prompt_specs))
        self.vision = TrtEngine(
            args.vision_engine,
            input_shapes={"images": (1, 3, args.image_size, args.image_size)},
            host_outputs=(),
        )
        self.text = TrtEngine(
            args.text_engine,
            input_shapes={
                "input_ids": (len(self.prompt_specs), 32),
                "attention_mask": (len(self.prompt_specs), 32),
            },
            host_outputs=(),
        )
        self.decoder = TrtEngine(
            args.decoder_engine,
            host_outputs=("pred_logits", "presence_logits", "pred_masks"),
        )
        self.stream = cuda.Stream()
        self.tokenizer = Tokenizer.from_file(args.tokenizer)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)
        self._prime_text()

    def _prime_text(self):
        input_ids = []
        attention_mask = []
        for _category, prompt in self.prompt_specs:
            encoded = self.tokenizer.encode(prompt)
            input_ids.append(encoded.ids)
            attention_mask.append(encoded.attention_mask)
        inputs = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
        }
        self.text.enqueue(inputs, self.stream)
        self.stream.synchronize()

    def preprocess_image(self, image):
        resized = image.resize(self.vision.shape["images"][-2:][::-1], resample=PIL.Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
        arr = arr / 127.5 - 1.0
        arr = arr.transpose(2, 0, 1)[None, :, :, :]
        return np.ascontiguousarray(arr.astype(np.float32))

    def bind_decoder_common(self):
        for name in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"):
            self.decoder.bind_from(name, self.vision, name)

    def copy_selected_masks(self, keep_indices, stream):
        out = self.decoder.hbuf["pred_masks"]
        mask_h, mask_w = out.shape[-2], out.shape[-1]
        itemsize = out.dtype.itemsize
        mask_nbytes = mask_h * mask_w * itemsize
        base = int(self.decoder.dptr["pred_masks"])
        for idx in keep_indices:
            cuda.memcpy_dtoh_async(out[0, int(idx)], base + int(idx) * mask_nbytes, stream)

    def decode_prompt(self, image, score_thr, mask_thr, process_area_thr, process_prefer):
        self.decoder.copy_output("pred_logits", self.stream)
        self.decoder.copy_output("presence_logits", self.stream)
        self.stream.synchronize()

        logits = self.decoder.hbuf["pred_logits"][0].astype(np.float32)
        presence = float(self.decoder.hbuf["presence_logits"][0, 0])
        scores_all = (sigmoid(logits) * float(sigmoid(presence))).astype(np.float32)
        keep = np.flatnonzero(scores_all > score_thr)
        raw_count = int(keep.size)
        if keep.size == 0:
            return (
                np.empty((0, image.height, image.width), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                raw_count,
            )

        self.copy_selected_masks(keep, self.stream)
        self.stream.synchronize()
        masks_logits = self.decoder.hbuf["pred_masks"][0, keep].copy()
        scores = scores_all[keep]

        masks_bool = np.empty((keep.size, image.height, image.width), dtype=bool)
        for i, mask_logit in enumerate(masks_logits):
            resized = cv2.resize(
                mask_logit.astype(np.float32),
                (image.width, image.height),
                interpolation=cv2.INTER_LINEAR,
            )
            masks_bool[i] = resized > mask_thr

        processed_masks, processed_indices = iitp.process_masks_by_size(
            masks_bool,
            threshold=process_area_thr,
            prefer=process_prefer,
        )
        scores = scores[processed_indices] if processed_indices.size > 0 else scores[:0]
        return processed_masks.astype(np.float32), scores.astype(np.float32), raw_count

    def infer_image(self, image, score_thr, mask_thr, process_area_thr, process_prefer):
        x = self.preprocess_image(image)
        prompt_results = {
            category: iitp.empty_prompt_result(image.height, image.width)
            for category in self.prompt_categories
        }

        self.vision.enqueue({"images": x}, self.stream)
        text_feat_stride = 32 * 256 * self.text.dtype["text_features"].itemsize
        text_mask_stride = 32 * self.text.dtype["text_mask"].itemsize
        for prompt_idx, (category, prompt) in enumerate(self.prompt_specs):
            self.bind_decoder_common()
            self.decoder.bind_from("prompt_features", self.text, "text_features", prompt_idx * text_feat_stride)
            self.decoder.bind_from("prompt_mask", self.text, "text_mask", prompt_idx * text_mask_stride)
            self.decoder.enqueue({}, self.stream)
            masks, scores, raw_count = self.decode_prompt(
                image,
                score_thr=score_thr,
                mask_thr=mask_thr,
                process_area_thr=process_area_thr,
                process_prefer=process_prefer,
            )
            iitp.append_prompt_result(
                prompt_results[category],
                {
                    "masks": masks,
                    "scores": scores,
                    "raw_count": raw_count,
                    "mask_nms_removed": 0,
                    "prompts": [prompt],
                },
            )
        self.stream.synchronize()
        return prompt_results


class TemporalMaskVoter:
    def __init__(
        self,
        categories,
        dx_px_per_frame=0.0,
        dy_px_per_frame=0.0,
        iou_thr=0.20,
        vote_thr=0.50,
        current_weight=1.0,
        history_weight=0.65,
        decay=0.85,
        max_age=2,
        mask_thr=0.5,
    ):
        self.categories = tuple(categories)
        self.dx_px_per_frame = float(dx_px_per_frame)
        self.dy_px_per_frame = float(dy_px_per_frame)
        self.iou_thr = float(iou_thr)
        self.vote_thr = float(vote_thr)
        self.current_weight = float(current_weight)
        self.history_weight = float(history_weight)
        self.decay = float(decay)
        self.max_age = int(max_age)
        self.mask_thr = float(mask_thr)
        self.tracks = {category: [] for category in self.categories}

    @staticmethod
    def _shift_mask(mask, dx, dy):
        height, width = mask.shape
        transform = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
        shifted = cv2.warpAffine(
            mask.astype(np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return shifted.astype(bool)

    @staticmethod
    def _mask_iou(a, b):
        inter = np.logical_and(a, b).sum(dtype=np.float64)
        union = np.logical_or(a, b).sum(dtype=np.float64)
        return float(inter / (union + 1e-6))

    def _empty_like(self, result):
        height, width = result["masks"].shape[-2:]
        return {
            "masks": np.empty((0, height, width), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "raw_count": int(result.get("raw_count", 0)),
            "mask_nms_removed": int(result.get("mask_nms_removed", 0)),
            "prompts": list(result.get("prompts", [])),
        }

    def update(self, prompt_results, frame_delta=1.0):
        dx = self.dx_px_per_frame * float(frame_delta)
        dy = self.dy_px_per_frame * float(frame_delta)
        updated = {}
        next_tracks = {}
        for category, result in prompt_results.items():
            predicted = []
            for track in self.tracks.get(category, []):
                pred_mask = self._shift_mask(track["mask"], dx, dy)
                if pred_mask.any():
                    predicted.append({
                        "mask": pred_mask,
                        "score": float(track["score"]) * (self.decay ** max(float(frame_delta), 1.0)),
                        "age": int(track["age"]) + 1,
                    })

            masks = result["masks"]
            scores = result["scores"]
            used_tracks = set()
            out_masks = []
            out_scores = []
            category_tracks = []

            order = np.argsort(-scores) if scores.size > 0 else np.empty((0,), dtype=np.int64)
            for det_idx in order:
                det_mask = masks[det_idx] >= self.mask_thr
                if not det_mask.any():
                    continue
                best_track = -1
                best_iou = 0.0
                for track_idx, track in enumerate(predicted):
                    if track_idx in used_tracks:
                        continue
                    iou = self._mask_iou(det_mask, track["mask"])
                    if iou > best_iou:
                        best_track = track_idx
                        best_iou = iou
                if best_track >= 0 and best_iou >= self.iou_thr:
                    track = predicted[best_track]
                    used_tracks.add(best_track)
                    vote = (
                        self.current_weight * det_mask.astype(np.float32)
                        + self.history_weight * track["mask"].astype(np.float32)
                    ) / max(self.current_weight + self.history_weight, 1e-6)
                    voted_mask = vote >= self.vote_thr
                    score = max(float(scores[det_idx]), float(track["score"]))
                else:
                    voted_mask = det_mask
                    score = float(scores[det_idx])
                out_masks.append(voted_mask.astype(np.float32))
                out_scores.append(score)
                category_tracks.append({"mask": voted_mask, "score": score, "age": 0})

            out = self._empty_like(result)
            if out_masks:
                out["masks"] = np.stack(out_masks, axis=0).astype(np.float32)
                out["scores"] = np.asarray(out_scores, dtype=np.float32)
            updated[category] = out
            next_tracks[category] = category_tracks[: self.max_age + 16]

        self.tracks = {category: next_tracks.get(category, []) for category in self.categories}
        return updated


def apply_precision_preset(args):
    args.transparent_min_score = 0.60
    args.metal_min_score = 0.50
    args.cardboard_min_score = 0.45
    args.cross_category_iou = 0.25
    args.cross_category_overlap = 0.55
    args.transparent_score_scale = 1.05
    args.metal_score_scale = 0.90
    args.cardboard_score_scale = 1.00


def resolve_category_conflicts(prompt_results, mask_thr, iou_thr, overlap_thr, score_scales):
    candidates = []
    for category, result in prompt_results.items():
        if category == getattr(iitp, "CONVEYOR_STEM", "conveyor_belt"):
            continue
        for mask_idx, score in enumerate(result["scores"]):
            mask_bool = result["masks"][mask_idx] >= mask_thr
            if mask_bool.any():
                candidates.append({
                    "category": category,
                    "score": float(score),
                    "rank_score": float(score) * float(score_scales.get(category, 1.0)),
                    "mask": mask_bool,
                })
    candidates.sort(key=lambda item: item["rank_score"], reverse=True)

    kept = []
    for cand in candidates:
        if any(iitp.masks_conflict(cand["mask"], prev["mask"], iou_thr, overlap_thr) for prev in kept):
            continue
        kept.append(cand)

    if not prompt_results:
        return
    sample = next(iter(prompt_results.values()))
    height, width = sample["masks"].shape[-2:]
    grouped_masks = {category: [] for category in prompt_results.keys()}
    grouped_scores = {category: [] for category in prompt_results.keys()}
    for item in kept:
        grouped_masks[item["category"]].append(item["mask"].astype(np.float32))
        grouped_scores[item["category"]].append(item["score"])
    for category, result in prompt_results.items():
        if category == getattr(iitp, "CONVEYOR_STEM", "conveyor_belt"):
            continue
        if grouped_masks[category]:
            result["masks"] = np.stack(grouped_masks[category], axis=0).astype(np.float32)
            result["scores"] = np.asarray(grouped_scores[category], dtype=np.float32)
        else:
            result["masks"] = np.empty((0, height, width), dtype=np.float32)
            result["scores"] = np.empty((0,), dtype=np.float32)


class RuntimeSegmenter:
    def __init__(self, args):
        self.args = args
        self.runner = UslsSam3Runner(args)
        self.lock = threading.Lock()
        self.frame_counter = 0
        self.temporal = None
        if args.temporal_vote:
            self.temporal = TemporalMaskVoter(
                categories=self.runner.prompt_categories,
                dx_px_per_frame=args.conveyor_dx_px_per_frame,
                dy_px_per_frame=args.conveyor_dy_px_per_frame,
                iou_thr=args.temporal_iou_thr,
                vote_thr=args.temporal_vote_thr,
                current_weight=args.temporal_current_weight,
                history_weight=args.temporal_history_weight,
                decay=args.temporal_decay,
                max_age=args.temporal_max_age,
                mask_thr=0.5,
            )

    def infer(self, image, filename="frame.jpg", frame_delta=1.0, save_overlay=False):
        t0 = time.perf_counter()
        with self.lock:
            prompt_results = self.runner.infer_image(
                image,
                score_thr=self.args.score_thr,
                mask_thr=self.args.mask_thr,
                process_area_thr=self.args.process_area_thr,
                process_prefer=self.args.process_prefer,
            )
            t1 = time.perf_counter()
            if self.temporal is not None:
                prompt_results = self.temporal.update(prompt_results, frame_delta=frame_delta)
            result = self._postprocess(image, filename, prompt_results, save_overlay=save_overlay)
            t2 = time.perf_counter()
        result["timing_ms"] = {
            "inference": 1000.0 * (t1 - t0),
            "postprocess": 1000.0 * (t2 - t1),
            "total": 1000.0 * (t2 - t0),
        }
        return result

    def _postprocess(self, image, filename, prompt_results, save_overlay=False):
        min_scores = {
            "transparent": self.args.transparent_min_score,
            "metal": self.args.metal_min_score,
            "Cardboard": self.args.cardboard_min_score,
        }
        for category, min_score in min_scores.items():
            if category in prompt_results:
                iitp.filter_prompt_result_by_score(prompt_results[category], min_score)

        resolve_category_conflicts(
            prompt_results,
            mask_thr=0.5,
            iou_thr=self.args.cross_category_iou,
            overlap_thr=self.args.cross_category_overlap,
            score_scales={
                "transparent": self.args.transparent_score_scale,
                "metal": self.args.metal_score_scale,
                "Cardboard": self.args.cardboard_score_scale,
            },
        )

        for category, result in prompt_results.items():
            if category == getattr(iitp, "CONVEYOR_STEM", "conveyor_belt"):
                continue
            iitp.nms_prompt_result_by_bbox(result, mask_thr=0.5, iou_thr=self.args.merge_nms_iou)

        height, width = image.height, image.width
        instances = []
        masks = []
        index_mask = np.zeros((height, width), dtype=np.uint16)
        for category, result in prompt_results.items():
            if category == getattr(iitp, "CONVEYOR_STEM", "conveyor_belt"):
                continue
            order = np.argsort(-result["scores"]) if result["scores"].size > 0 else np.empty((0,), dtype=np.int64)
            if self.args.top_k is not None:
                order = order[: self.args.top_k]
            for mask_idx in order:
                mask_bool = result["masks"][mask_idx] >= 0.5
                box = iitp.bbox_from_mask(mask_bool)
                if box is None:
                    continue
                inst_id = len(instances) + 1
                index_mask[mask_bool] = inst_id
                masks.append(mask_bool.astype(np.uint8))
                instances.append({
                    "id": inst_id,
                    "category": category,
                    "category_id": int(CATEGORY_TO_ID.get(category, 0)),
                    "score": float(result["scores"][mask_idx]),
                    "bbox_xyxy": [float(x) for x in box.tolist()],
                })

        mask_stack = (
            np.stack(masks, axis=0).astype(np.uint8)
            if masks else np.zeros((0, height, width), dtype=np.uint8)
        )
        if save_overlay:
            os.makedirs(self.args.out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(filename))[0]
            overlay_path = os.path.join(self.args.out_dir, f"{stem}_prompts.png")
            overlay = iitp.render_prompt_color_overlay(image, prompt_results, alpha=0.50, mask_thr=0.5)
            if isinstance(overlay, np.ndarray):
                overlay = PIL.Image.fromarray(overlay.astype(np.uint8))
            overlay.save(overlay_path)
        else:
            overlay_path = None
        return {
            "filename": os.path.basename(filename),
            "height": int(height),
            "width": int(width),
            "instances": instances,
            "index_mask": index_mask,
            "mask_stack": mask_stack,
            "overlay_path": overlay_path,
        }


def serve_tcp(args, segmenter):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[usls tcp] listening on {args.host}:{args.port}")
    while True:
        conn, addr = server.accept()
        print(f"[usls tcp] client connected: {addr}")
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        try:
            while True:
                meta, payload = recv_packet(conn)
                if meta.get("type") in {"close", "shutdown"}:
                    break
                image, filename = decode_image_packet(meta, payload)
                frame_delta = float(meta.get("frame_delta", 1.0))
                result = segmenter.infer(
                    image,
                    filename=filename,
                    frame_delta=frame_delta,
                    save_overlay=args.save_overlays,
                )
                arrays = {
                    "index_mask": result["index_mask"],
                    "category_ids": np.asarray(
                        [inst["category_id"] for inst in result["instances"]],
                        dtype=np.int32,
                    ),
                    "scores": np.asarray(
                        [inst["score"] for inst in result["instances"]],
                        dtype=np.float32,
                    ),
                    "boxes_xyxy": np.asarray(
                        [inst["bbox_xyxy"] for inst in result["instances"]],
                        dtype=np.float32,
                    ).reshape((-1, 4)),
                }
                if args.reply_mask_stack:
                    arrays["mask_stack"] = result["mask_stack"]
                reply_meta, reply_payload = pack_arrays(
                    arrays,
                    "seg_result",
                    {
                        "idx": meta.get("idx", -1),
                        "filename": result["filename"],
                        "height": result["height"],
                        "width": result["width"],
                        "instances": result["instances"],
                        "timing_ms": result["timing_ms"],
                        "category_to_id": CATEGORY_TO_ID,
                        "overlay_path": result["overlay_path"],
                    },
                )
                send_packet(conn, reply_meta, reply_payload)
                if not args.quiet:
                    print(
                        f"[usls tcp] {result['filename']} "
                        f"instances={len(result['instances'])} "
                        f"total={result['timing_ms']['total']:.1f}ms"
                    )
        except ConnectionError:
            pass
        except Exception as exc:
            err = {"type": "error", "message": str(exc)}
            try:
                send_packet(conn, err, b"")
            except Exception:
                pass
            print(f"[usls tcp] client error: {exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print("[usls tcp] client disconnected")


def serve_ros2(args, segmenter):
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
    except Exception as exc:
        raise RuntimeError("ROS2 mode requires rclpy, sensor_msgs, std_msgs, and cv_bridge") from exc

    bridge = CvBridge()

    class UslsNode(Node):
        def __init__(self):
            super().__init__("usls_sam3_trt_server")
            self.pub_json = self.create_publisher(String, args.ros_result_topic, 10)
            self.pub_mask = self.create_publisher(Image, args.ros_mask_topic, 10)
            self.sub = self.create_subscription(Image, args.ros_image_topic, self.on_image, 10)
            self.frame_idx = 0

        def on_image(self, msg):
            self.frame_idx += 1
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil = PIL.Image.fromarray(cv_img)
            filename = f"{self.frame_idx:06d}.jpg"
            result = segmenter.infer(pil, filename=filename, frame_delta=1.0, save_overlay=args.save_overlays)
            out = String()
            out.data = json.dumps({
                "filename": result["filename"],
                "height": result["height"],
                "width": result["width"],
                "instances": result["instances"],
                "timing_ms": result["timing_ms"],
            })
            self.pub_json.publish(out)
            mask_msg = bridge.cv2_to_imgmsg(result["index_mask"], encoding="mono16")
            mask_msg.header = msg.header
            self.pub_mask.publish(mask_msg)
            if not args.quiet:
                self.get_logger().info(
                    f"{result['filename']} instances={len(result['instances'])} "
                    f"total={result['timing_ms']['total']:.1f}ms"
                )

    rclpy.init()
    node = UslsNode()
    print(f"[usls ros2] subscribing {args.ros_image_topic}, publishing {args.ros_result_topic}")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_prompt_specs(items):
    if not items:
        return DEFAULT_PROMPT_SPECS
    specs = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--prompt-spec must be category=prompt, got: {item}")
        category, prompt = item.split("=", 1)
        specs.append((category.strip(), prompt.strip()))
    return tuple(specs)


def parse_args():
    parser = argparse.ArgumentParser(description="Runtime TCP/ROS2 server for IITP USLS SAM3 TensorRT segmentation.")
    parser.add_argument("--mode", choices=("tcp", "ros2"), default="tcp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6020)
    parser.add_argument("--ros-image-topic", default="/camera/color/image_raw")
    parser.add_argument("--ros-result-topic", default="/usls/segmentation_json")
    parser.add_argument("--ros-mask-topic", default="/usls/index_mask")
    parser.add_argument("--vision-engine", default=os.path.join(REPO_ROOT, "weights", "SAM3-trt", "usls_engines_b2p2", "vision_b2_fp16.engine"))
    parser.add_argument("--text-engine", default=os.path.join(REPO_ROOT, "weights", "SAM3-trt", "usls_engines", "text_b6_fp16.engine"))
    parser.add_argument("--decoder-engine", default=os.path.join(REPO_ROOT, "weights", "SAM3-trt", "usls_engines", "decoder_b1_p32_fp16.engine"))
    parser.add_argument("--tokenizer", default=os.path.join(REPO_ROOT, "weights", "usls", "tokenizer.json"))
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--score-thr", type=float, default=0.4)
    parser.add_argument("--mask-thr", type=float, default=0.0)
    parser.add_argument("--process-area-thr", type=int, default=1000)
    parser.add_argument("--process-prefer", choices=("smaller", "larger"), default="smaller")
    parser.add_argument("--transparent-min-score", type=float, default=0.60)
    parser.add_argument("--metal-min-score", type=float, default=0.50)
    parser.add_argument("--cardboard-min-score", type=float, default=0.45)
    parser.add_argument("--merge-nms-iou", type=float, default=0.5)
    parser.add_argument("--cross-category-iou", type=float, default=0.25)
    parser.add_argument("--cross-category-overlap", type=float, default=0.55)
    parser.add_argument("--transparent-score-scale", type=float, default=1.05)
    parser.add_argument("--metal-score-scale", type=float, default=0.90)
    parser.add_argument("--cardboard-score-scale", type=float, default=1.00)
    parser.add_argument("--precision-tuned-preset", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temporal-vote", action="store_true")
    parser.add_argument("--conveyor-dx-px-per-frame", type=float, default=-1.0)
    parser.add_argument("--conveyor-dy-px-per-frame", type=float, default=-68.0)
    parser.add_argument("--temporal-iou-thr", type=float, default=0.20)
    parser.add_argument("--temporal-vote-thr", type=float, default=0.50)
    parser.add_argument("--temporal-current-weight", type=float, default=1.0)
    parser.add_argument("--temporal-history-weight", type=float, default=0.65)
    parser.add_argument("--temporal-decay", type=float, default=0.85)
    parser.add_argument("--temporal-max-age", type=int, default=2)
    parser.add_argument("--reply-mask-stack", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "output", "usls_runtime_overlays"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--prompt-spec", action="append", default=None)
    args = parser.parse_args()
    args.prompt_specs = parse_prompt_specs(args.prompt_spec)
    if args.precision_tuned_preset:
        apply_precision_preset(args)
    return args


def main():
    args = parse_args()
    segmenter = RuntimeSegmenter(args)
    if args.mode == "tcp":
        serve_tcp(args, segmenter)
    else:
        serve_ros2(args, segmenter)


if __name__ == "__main__":
    main()
