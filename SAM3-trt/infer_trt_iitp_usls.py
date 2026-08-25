import argparse
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np
import PIL.Image
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from tokenizers import Tokenizer


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "iitp_common",
    os.path.join(ROOT, "SAM3-trt", "infer_trt_iitp.py"),
)
iitp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iitp)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

DEFAULT_PROMPT_SPECS = (
    ("transparent", "transparent plastic bottle"),
    ("metal", "beverage can"),
    ("metal", "aluminum foil wrapper"),
    ("metal", "foil packaging"),
    ("Cardboard", "cardboard box"),
    ("Cardboard", "cardboard package"),
)


def _parse_numeric_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def trt_dtype_to_np(dtype):
    return np.dtype(trt.nptype(dtype))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


class TrtEngine:
    def __init__(self, path, input_shapes=None, host_outputs=()):
        self.path = path
        self.stream = cuda.Stream()
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {path}")

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
                    raise ValueError(f"Dynamic input shape for {name} must be provided: {path}")
                self.context.set_input_shape(name, shape)

        self.allocate(host_outputs=host_outputs)

    def tensor_shape(self, name):
        if self.is_input[name]:
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
        else:
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
            input_shapes={"images": (1, 3, 1008, 1008)},
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

    @staticmethod
    def preprocess_image(image):
        resized = image.resize((1008, 1008), resample=PIL.Image.BILINEAR)
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
            resized = cv2.resize(mask_logit.astype(np.float32), (image.width, image.height), interpolation=cv2.INTER_LINEAR)
            masks_bool[i] = resized > mask_thr

        processed_masks, processed_indices = iitp.process_masks_by_size(
            masks_bool,
            threshold=process_area_thr,
            prefer=process_prefer,
        )
        scores = scores[processed_indices] if processed_indices.size > 0 else scores[:0]
        return processed_masks.astype(np.float32), scores.astype(np.float32), raw_count

    def infer_image(
        self,
        image,
        score_thr=0.5,
        mask_thr=0.0,
        process_area_thr=1000,
        process_prefer="smaller",
        profile=False,
    ):
        x = self.preprocess_image(image)
        prompt_results = {
            category: iitp.empty_prompt_result(image.height, image.width)
            for category in self.prompt_categories
        }

        e0 = cuda.Event()
        e1 = cuda.Event()
        e2 = cuda.Event()
        e3 = cuda.Event()
        e0.record(self.stream)
        self.vision.enqueue({"images": x}, self.stream)
        e1.record(self.stream)

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

        e2.record(self.stream)
        e2.synchronize()
        prof = {}
        if profile:
            prof = {
                "vision": float(e0.time_till(e1)),
                "decoder_total": float(e1.time_till(e2)),
                "total_cuda": float(e0.time_till(e2)),
            }
        return prompt_results, prof


class TemporalMaskVoter:
    """Conveyor-aware temporal voting over per-frame detector masks."""

    def __init__(
        self,
        categories,
        dx_px_per_frame=0.0,
        dy_px_per_frame=0.0,
        delta_mode="order",
        iou_thr=0.20,
        vote_thr=0.50,
        current_weight=1.0,
        history_weight=0.65,
        decay=0.85,
        max_age=2,
        inject_missed=True,
        missed_min_score=0.35,
        mask_thr=0.5,
    ):
        self.categories = tuple(categories)
        self.dx_px_per_frame = float(dx_px_per_frame)
        self.dy_px_per_frame = float(dy_px_per_frame)
        self.delta_mode = str(delta_mode)
        self.iou_thr = float(iou_thr)
        self.vote_thr = float(vote_thr)
        self.current_weight = float(current_weight)
        self.history_weight = float(history_weight)
        self.decay = float(decay)
        self.max_age = int(max_age)
        self.inject_missed = bool(inject_missed)
        self.missed_min_score = float(missed_min_score)
        self.mask_thr = float(mask_thr)
        self.tracks = {category: [] for category in self.categories}
        self.prev_order_idx = None
        self.prev_stem_idx = None

    @staticmethod
    def _empty_like(result):
        height, width = result["masks"].shape[-2:]
        return {
            "masks": np.empty((0, height, width), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "raw_count": 0,
            "mask_nms_removed": 0,
            "prompts": list(result.get("prompts", [])),
        }

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

    def _frame_delta(self, order_idx, path):
        if self.prev_order_idx is None:
            return 0.0
        if self.delta_mode == "stem":
            cur_stem = _parse_numeric_stem(path)
            if cur_stem is not None and self.prev_stem_idx is not None:
                return float(cur_stem - self.prev_stem_idx)
        return float(order_idx - self.prev_order_idx)

    def _predict_tracks(self, frame_delta):
        dx = self.dx_px_per_frame * frame_delta
        dy = self.dy_px_per_frame * frame_delta
        predicted = {}
        for category, tracks in self.tracks.items():
            predicted_tracks = []
            for track in tracks:
                pred_mask = self._shift_mask(track["mask"], dx, dy) if frame_delta != 0.0 else track["mask"].copy()
                if pred_mask.any():
                    predicted_tracks.append({
                        "mask": pred_mask,
                        "score": float(track["score"]) * (self.decay ** max(frame_delta, 1.0)),
                        "age": int(track["age"]) + 1,
                    })
            predicted[category] = predicted_tracks
        return predicted

    def _vote_category(self, result, predicted_tracks):
        out = self._empty_like(result)
        out["raw_count"] = int(result.get("raw_count", 0))
        out["mask_nms_removed"] = int(result.get("mask_nms_removed", 0))

        masks = result["masks"]
        scores = result["scores"]
        used_tracks = set()
        new_tracks = []
        out_masks = []
        out_scores = []

        order = np.argsort(-scores) if scores.size > 0 else np.empty((0,), dtype=np.int64)
        for det_idx in order:
            det_mask = masks[det_idx] >= self.mask_thr
            if not det_mask.any():
                continue

            best_track = -1
            best_iou = 0.0
            for track_idx, track in enumerate(predicted_tracks):
                if track_idx in used_tracks:
                    continue
                iou = self._mask_iou(det_mask, track["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_idx

            if best_track >= 0 and best_iou >= self.iou_thr:
                track = predicted_tracks[best_track]
                used_tracks.add(best_track)
                vote = (
                    self.current_weight * det_mask.astype(np.float32)
                    + self.history_weight * track["mask"].astype(np.float32)
                ) / max(self.current_weight + self.history_weight, 1e-6)
                voted_mask = vote >= self.vote_thr
                if not voted_mask.any():
                    voted_mask = det_mask
                voted_score = max(float(scores[det_idx]), float(track["score"]))
                new_tracks.append({"mask": voted_mask, "score": voted_score, "age": 0})
                out_masks.append(voted_mask.astype(np.float32))
                out_scores.append(voted_score)
            else:
                det_score = float(scores[det_idx])
                new_tracks.append({"mask": det_mask, "score": det_score, "age": 0})
                out_masks.append(det_mask.astype(np.float32))
                out_scores.append(det_score)

        if self.inject_missed:
            for track_idx, track in enumerate(predicted_tracks):
                if track_idx in used_tracks:
                    continue
                if int(track["age"]) > self.max_age or float(track["score"]) < self.missed_min_score:
                    continue
                track_mask = track["mask"]
                if not track_mask.any():
                    continue
                new_tracks.append({
                    "mask": track_mask,
                    "score": float(track["score"]),
                    "age": int(track["age"]),
                })
                out_masks.append(track_mask.astype(np.float32))
                out_scores.append(float(track["score"]))

        if out_masks:
            out["masks"] = np.stack(out_masks, axis=0).astype(np.float32)
            out["scores"] = np.asarray(out_scores, dtype=np.float32)
        self._next_tracks = new_tracks
        return out

    def update(self, prompt_results, order_idx, path):
        frame_delta = self._frame_delta(order_idx, path)
        predicted = self._predict_tracks(frame_delta)
        updated = {}
        next_tracks = {}
        for category, result in prompt_results.items():
            self._next_tracks = []
            updated[category] = self._vote_category(result, predicted.get(category, []))
            next_tracks[category] = self._next_tracks

        self.tracks = {category: next_tracks.get(category, []) for category in self.categories}
        self.prev_order_idx = int(order_idx)
        self.prev_stem_idx = _parse_numeric_stem(path)
        return updated


def _parse_roi_arg(value, width, height):
    if value is None or str(value).strip() == "":
        return int(width * 0.08), 0, int(width * 0.64), int(height)
    parts = [int(round(float(x))) for x in str(value).replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("--motion-init-roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def estimate_conveyor_motion_from_images(
    paths,
    num_frames=10,
    roi_arg=None,
    direction="up",
    min_mag=8.0,
    max_mag=90.0,
):
    selected = list(paths[: max(2, int(num_frames))])
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in selected]
    images = [img for img in images if img is not None]
    if len(images) < 2:
        return None

    height, width = images[0].shape[:2]
    x0, y0, x1, y1 = _parse_roi_arg(roi_arg, width, height)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=8, edgeThreshold=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    all_disp = []
    pair_estimates = []

    def direction_ok(delta):
        dx, dy = float(delta[0]), float(delta[1])
        if direction == "up":
            return dy < -5.0
        if direction == "down":
            return dy > 5.0
        if direction == "left":
            return dx < -5.0
        if direction == "right":
            return dx > 5.0
        return True

    for prev_img, cur_img in zip(images, images[1:]):
        prev_gray = cv2.cvtColor(prev_img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        key_prev, desc_prev = orb.detectAndCompute(prev_gray, None)
        key_cur, desc_cur = orb.detectAndCompute(cur_gray, None)
        if desc_prev is None or desc_cur is None:
            continue
        matches = matcher.knnMatch(desc_prev, desc_cur, k=2)
        disps = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance > 0.75 * n.distance:
                continue
            p = np.asarray(key_prev[m.queryIdx].pt, dtype=np.float32)
            q = np.asarray(key_cur[m.trainIdx].pt, dtype=np.float32)
            disp = q - p
            mag = float(np.linalg.norm(disp))
            if min_mag <= mag <= max_mag and direction_ok(disp):
                disps.append(disp)
        if not disps:
            continue
        disps = np.asarray(disps, dtype=np.float32)
        median = np.median(disps, axis=0)
        err = np.linalg.norm(disps - median, axis=1)
        keep = err < 18.0
        robust = disps[keep] if keep.any() else disps
        pair_median = np.median(robust, axis=0)
        pair_estimates.append(pair_median)
        all_disp.extend(robust)

    if not all_disp:
        return None
    all_disp = np.asarray(all_disp, dtype=np.float32)
    median = np.median(all_disp, axis=0)
    q25 = np.percentile(all_disp, 25, axis=0)
    q75 = np.percentile(all_disp, 75, axis=0)
    return {
        "dx": float(median[0]),
        "dy": float(median[1]),
        "q25": [float(q25[0]), float(q25[1])],
        "q75": [float(q75[0]), float(q75[1])],
        "matches": int(all_disp.shape[0]),
        "pairs": int(len(pair_estimates)),
        "roi_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "direction": direction,
    }


def clone_prompt_results(prompt_results):
    cloned = {}
    for category, result in prompt_results.items():
        cloned[category] = {
            "masks": result["masks"].copy(),
            "scores": result["scores"].copy(),
            "raw_count": int(result.get("raw_count", 0)),
            "mask_nms_removed": int(result.get("mask_nms_removed", 0)),
            "prompts": list(result.get("prompts", [])),
        }
    return cloned


def resolve_category_conflicts(
    prompt_results,
    mask_thr=0.5,
    iou_thr=0.3,
    overlap_thr=0.6,
    score_scales=None,
):
    score_scales = score_scales or {}
    candidates = []
    for category, result in prompt_results.items():
        if category == getattr(iitp, "CONVEYOR_STEM", "conveyor_belt"):
            continue
        masks = result["masks"]
        scores = result["scores"]
        for mask_idx, score in enumerate(scores):
            mask_bool = masks[mask_idx] >= mask_thr
            if not mask_bool.any():
                continue
            candidates.append({
                "category": category,
                "score": float(score),
                "rank_score": float(score) * float(score_scales.get(category, 1.0)),
                "mask": mask_bool,
            })
    candidates.sort(key=lambda item: item["rank_score"], reverse=True)

    kept = []
    removed = {}
    for cand in candidates:
        if any(iitp.masks_conflict(cand["mask"], prev["mask"], iou_thr, overlap_thr) for prev in kept):
            removed[cand["category"]] = int(removed.get(cand["category"], 0)) + 1
            continue
        kept.append(cand)

    height = width = None
    for result in prompt_results.values():
        if result["masks"].size > 0:
            height, width = result["masks"].shape[1:]
            break
    if height is None or width is None:
        return removed

    target_categories = [
        category for category in prompt_results.keys()
        if category != getattr(iitp, "CONVEYOR_STEM", "conveyor_belt")
    ]
    grouped_masks = {category: [] for category in target_categories}
    grouped_scores = {category: [] for category in target_categories}
    for item in kept:
        grouped_masks[item["category"]].append(item["mask"].astype(np.float32))
        grouped_scores[item["category"]].append(item["score"])

    for category in target_categories:
        if grouped_masks[category]:
            prompt_results[category]["masks"] = np.stack(grouped_masks[category], axis=0).astype(np.float32)
            prompt_results[category]["scores"] = np.asarray(grouped_scores[category], dtype=np.float32)
        else:
            prompt_results[category]["masks"] = np.empty((0, height, width), dtype=np.float32)
            prompt_results[category]["scores"] = np.empty((0,), dtype=np.float32)
        prompt_results[category]["mask_nms_removed"] = int(prompt_results[category].get("mask_nms_removed", 0)) + int(
            removed.get(category, 0)
        )
    return removed


def finalize_usls_prompt_results_row(
    image,
    image_path,
    prompt_results,
    out_dir,
    category_min_scores,
    merge_nms_iou,
    mask_thr,
    top_k,
    mask_nms_iou,
    mask_nms_overlap,
    save_overlay,
    quiet,
    cross_category_mask_nms=False,
    cross_category_iou=0.3,
    cross_category_overlap=0.6,
    category_score_scales=None,
):
    category_score_scales = category_score_scales or {}
    if not cross_category_mask_nms:
        return iitp.finalize_prompt_results_row(
            image=image,
            image_path=image_path,
            prompt_results=prompt_results,
            out_dir=out_dir,
            category_min_scores=category_min_scores,
            merge_nms_iou=merge_nms_iou,
            mask_thr=mask_thr,
            top_k=top_k,
            suppress_conveyor=False,
            mask_nms=True,
            mask_nms_iou=mask_nms_iou,
            mask_nms_overlap=mask_nms_overlap,
            save_overlay=save_overlay,
            quiet=quiet,
        )

    for category, min_score in category_min_scores.items():
        if category in prompt_results:
            iitp.filter_prompt_result_by_score(prompt_results[category], min_score)
    removed = resolve_category_conflicts(
        prompt_results,
        mask_thr=mask_thr,
        iou_thr=cross_category_iou,
        overlap_thr=cross_category_overlap,
        score_scales=category_score_scales,
    )
    zero_thresholds = {category: 0.0 for category in category_min_scores.keys()}
    row = iitp.finalize_prompt_results_row(
        image=image,
        image_path=image_path,
        prompt_results=prompt_results,
        out_dir=out_dir,
        category_min_scores=zero_thresholds,
        merge_nms_iou=merge_nms_iou,
        mask_thr=mask_thr,
        top_k=top_k,
        suppress_conveyor=False,
        mask_nms=True,
        mask_nms_iou=mask_nms_iou,
        mask_nms_overlap=mask_nms_overlap,
        save_overlay=save_overlay,
        quiet=quiet,
    )
    row["category_min_scores"] = {k: float(v) for k, v in category_min_scores.items()}
    row["cross_category_mask_nms"] = {
        "enabled": True,
        "iou_thr": float(cross_category_iou),
        "overlap_thr": float(cross_category_overlap),
        "score_scales": {k: float(v) for k, v in category_score_scales.items()},
        "removed": {k: int(v) for k, v in removed.items()},
    }
    return row


def tune_precision_from_records(records, gt_by_file, args, base_min_scores):
    transparent_grid = [0.50, 0.60, 0.70, 0.78]
    metal_grid = [0.50, 0.60, 0.70, 0.78]
    cardboard_grid = [0.35, 0.45, 0.55, 0.65]
    conflict_grid = [
        (False, args.cross_category_iou, args.cross_category_overlap, 1.0, 1.0),
        (True, 0.25, 0.55, 1.05, 0.90),
        (True, 0.30, 0.60, 1.05, 0.90),
        (True, 0.30, 0.60, 1.10, 0.85),
    ]

    best = None
    candidates_checked = 0
    for transparent_thr in transparent_grid:
        for metal_thr in metal_grid:
            for cardboard_thr in cardboard_grid:
                min_scores = {
                    "transparent": transparent_thr,
                    "metal": metal_thr,
                    "Cardboard": cardboard_thr,
                }
                for cross_enabled, cross_iou, cross_overlap, transparent_scale, metal_scale in conflict_grid:
                    candidates_checked += 1
                    rows = []
                    for rec in records:
                        row = finalize_usls_prompt_results_row(
                            image=rec["image"],
                            image_path=rec["path"],
                            prompt_results=clone_prompt_results(rec["prompt_results"]),
                            out_dir=args.out_dir,
                            category_min_scores=min_scores,
                            merge_nms_iou=args.merge_nms_iou,
                            mask_thr=0.5,
                            top_k=args.top_k,
                            mask_nms_iou=args.mask_nms_iou,
                            mask_nms_overlap=args.mask_nms_overlap,
                            save_overlay=False,
                            quiet=True,
                            cross_category_mask_nms=cross_enabled,
                            cross_category_iou=cross_iou,
                            cross_category_overlap=cross_overlap,
                            category_score_scales={
                                "transparent": transparent_scale,
                                "metal": metal_scale,
                                "Cardboard": 1.0,
                            },
                        )
                        rows.append(row)
                    metrics = iitp.evaluate_predictions(
                        gt_by_file,
                        rows,
                        iou_thr=args.eval_iou_thr,
                        y_min=None if args.no_eval_y_filter else args.crop_y_min,
                        y_max=None if args.no_eval_y_filter else args.crop_y_max,
                        y_filter_mode=args.crop_y_mode,
                    )
                    micro = metrics["micro"]
                    recall = float(micro["recall"])
                    precision = float(micro["precision"])
                    f1 = float(micro["f1"])
                    meets_recall = recall >= float(args.tune_recall_floor)
                    key = (
                        1 if meets_recall else 0,
                        precision if meets_recall else recall,
                        f1,
                        recall,
                    )
                    if best is None or key > best["key"]:
                        best = {
                            "key": key,
                            "min_scores": min_scores,
                            "cross_category_mask_nms": {
                                "enabled": bool(cross_enabled),
                                "iou_thr": float(cross_iou),
                                "overlap_thr": float(cross_overlap),
                                "score_scales": {
                                    "transparent": float(transparent_scale),
                                    "metal": float(metal_scale),
                                    "Cardboard": 1.0,
                                },
                            },
                            "metrics": metrics,
                            "meets_recall_floor": bool(meets_recall),
                        }
    if best is None:
        return {}
    best.pop("key", None)
    best["candidates_checked"] = int(candidates_checked)
    best["recall_floor"] = float(args.tune_recall_floor)
    best["base_min_scores"] = {k: float(v) for k, v in base_min_scores.items()}
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="IITP SAM3 USLS split TensorRT inference.")
    parser.add_argument("--image-dir", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "images"))
    parser.add_argument("--annotations", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "annotations_crop.jsonl"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "output", "iitp_usls_run"))
    parser.add_argument("--vision-engine", default=os.path.join(ROOT, "weights", "SAM3-trt", "usls_engines_b2p2", "vision_b2_fp16.engine"))
    parser.add_argument("--text-engine", default=os.path.join(ROOT, "weights", "SAM3-trt", "usls_engines", "text_b6_fp16.engine"))
    parser.add_argument("--decoder-engine", default=os.path.join(ROOT, "weights", "SAM3-trt", "usls_engines", "decoder_b1_p32_fp16.engine"))
    parser.add_argument("--tokenizer", default=os.path.join(ROOT, "weights", "usls", "tokenizer.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.4)
    parser.add_argument("--mask-thr", type=float, default=0.0)
    parser.add_argument("--process-area-thr", type=int, default=1000)
    parser.add_argument("--process-prefer", choices=("smaller", "larger"), default="smaller")
    parser.add_argument("--transparent-min-score", type=float, default=0.7)
    parser.add_argument("--metal-min-score", type=float, default=0.5)
    parser.add_argument("--cardboard-min-score", type=float, default=0.75)
    parser.add_argument("--merge-nms-iou", type=float, default=0.5)
    parser.add_argument("--mask-nms-iou", type=float, default=0.3)
    parser.add_argument("--mask-nms-overlap", type=float, default=0.6)
    parser.add_argument("--cross-category-mask-nms", action="store_true", help="Resolve overlapping masks across categories by calibrated score.")
    parser.add_argument("--cross-category-iou", type=float, default=0.30)
    parser.add_argument("--cross-category-overlap", type=float, default=0.60)
    parser.add_argument("--transparent-score-scale", type=float, default=1.0)
    parser.add_argument("--metal-score-scale", type=float, default=1.0)
    parser.add_argument("--cardboard-score-scale", type=float, default=1.0)
    parser.add_argument("--eval-iou-thr", type=float, default=0.5)
    parser.add_argument("--crop-y-min", type=float, default=40.0)
    parser.add_argument("--crop-y-max", type=float, default=440.0)
    parser.add_argument("--crop-y-mode", choices=("center", "overlap"), default="center")
    parser.add_argument("--no-eval-y-filter", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-save-overlays", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--precision-tuned-preset", action="store_true", help="Use the IITP validation-tuned precision/recall postprocess preset.")
    parser.add_argument("--tune-precision", action="store_true", help="Grid-search thresholds/conflict suppression, preferring precision once recall is high enough.")
    parser.add_argument("--tune-recall-floor", type=float, default=0.90)
    parser.add_argument("--temporal-vote", action="store_true", help="Enable conveyor-aware temporal mask voting.")
    parser.add_argument("--conveyor-dx-px-per-frame", type=float, default=0.0, help="Expected object motion in image x per frame/stem step.")
    parser.add_argument("--conveyor-dy-px-per-frame", type=float, default=0.0, help="Expected object motion in image y per frame/stem step.")
    parser.add_argument("--temporal-delta-mode", choices=("order", "stem"), default="order", help="Use sorted image order or numeric filename stem for motion delta.")
    parser.add_argument("--temporal-iou-thr", type=float, default=0.20, help="IoU threshold for matching current detections to shifted previous tracks.")
    parser.add_argument("--temporal-vote-thr", type=float, default=0.50, help="Threshold applied after current/history mask voting.")
    parser.add_argument("--temporal-current-weight", type=float, default=1.0, help="Vote weight for current-frame detector mask.")
    parser.add_argument("--temporal-history-weight", type=float, default=0.65, help="Vote weight for shifted previous mask.")
    parser.add_argument("--temporal-decay", type=float, default=0.85, help="Score decay per frame for unmatched/predicted tracks.")
    parser.add_argument("--temporal-max-age", type=int, default=2, help="Maximum age for carrying missed tracks.")
    parser.add_argument("--temporal-missed-min-score", type=float, default=0.35, help="Minimum decayed score for injecting a missed shifted track.")
    parser.add_argument("--no-temporal-inject-missed", dest="temporal_inject_missed", action="store_false", help="Do not add shifted previous tracks when detector misses them.")
    parser.add_argument("--auto-init-conveyor-motion", action="store_true", help="Estimate temporal motion from the first frames before running inference.")
    parser.add_argument("--motion-init-frames", type=int, default=10, help="Number of leading images used for automatic conveyor motion initialization.")
    parser.add_argument("--motion-init-roi", default=None, help="Optional motion-estimation ROI as x0,y0,x1,y1. Default uses the left conveyor band.")
    parser.add_argument("--motion-init-direction", choices=("up", "down", "left", "right", "any"), default="up", help="Expected conveyor direction used to reject feature-match outliers.")
    parser.set_defaults(temporal_inject_missed=True)
    parser.add_argument(
        "--prompt-spec",
        action="append",
        default=None,
        help="Prompt spec as category=prompt. Repeat to override defaults.",
    )
    args = parser.parse_args()
    if args.prompt_spec:
        specs = []
        for item in args.prompt_spec:
            if "=" not in item:
                raise ValueError(f"--prompt-spec must be category=prompt, got: {item}")
            category, prompt = item.split("=", 1)
            specs.append((category.strip(), prompt.strip()))
        args.prompt_specs = tuple(specs)
    else:
        args.prompt_specs = DEFAULT_PROMPT_SPECS
    if args.precision_tuned_preset:
        args.transparent_min_score = 0.60
        args.metal_min_score = 0.50
        args.cardboard_min_score = 0.45
        args.cross_category_mask_nms = True
        args.cross_category_iou = 0.25
        args.cross_category_overlap = 0.55
        args.transparent_score_scale = 1.05
        args.metal_score_scale = 0.90
        args.cardboard_score_scale = 1.00
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = iitp.collect_image_paths(args.image_dir, args.limit)
    if not paths:
        raise FileNotFoundError(f"No images found: {args.image_dir}")

    runner = UslsSam3Runner(args)
    min_scores = {
        "transparent": args.transparent_min_score,
        "metal": args.metal_min_score,
        "Cardboard": args.cardboard_min_score,
    }
    motion_init = None
    if args.auto_init_conveyor_motion:
        motion_init = estimate_conveyor_motion_from_images(
            paths,
            num_frames=args.motion_init_frames,
            roi_arg=args.motion_init_roi,
            direction=args.motion_init_direction,
        )
        if motion_init is None:
            print("[temporal vote] auto motion init failed; using CLI dx/dy values")
        else:
            args.conveyor_dx_px_per_frame = motion_init["dx"]
            args.conveyor_dy_px_per_frame = motion_init["dy"]
            print(
                "[temporal vote] auto motion init "
                f"dx={motion_init['dx']:.3f} dy={motion_init['dy']:.3f} "
                f"matches={motion_init['matches']} pairs={motion_init['pairs']} "
                f"roi={motion_init['roi_xyxy']} "
                f"q25={motion_init['q25']} q75={motion_init['q75']}"
            )
    temporal_voter = None
    if args.temporal_vote:
        temporal_voter = TemporalMaskVoter(
            categories=runner.prompt_categories,
            dx_px_per_frame=args.conveyor_dx_px_per_frame,
            dy_px_per_frame=args.conveyor_dy_px_per_frame,
            delta_mode=args.temporal_delta_mode,
            iou_thr=args.temporal_iou_thr,
            vote_thr=args.temporal_vote_thr,
            current_weight=args.temporal_current_weight,
            history_weight=args.temporal_history_weight,
            decay=args.temporal_decay,
            max_age=args.temporal_max_age,
            inject_missed=args.temporal_inject_missed,
            missed_min_score=args.temporal_missed_min_score,
            mask_thr=0.5,
        )
        if not args.quiet:
            print(
                "[temporal vote] enabled "
                f"dx={args.conveyor_dx_px_per_frame:.3f} dy={args.conveyor_dy_px_per_frame:.3f} "
                f"delta={args.temporal_delta_mode} iou_thr={args.temporal_iou_thr:.3f} "
                f"inject_missed={bool(args.temporal_inject_missed)}"
            )

    rows = []
    timings = []
    tune_records = []
    pred_jsonl_path = os.path.join(args.out_dir, "predictions.jsonl")
    for idx, path in enumerate(paths, start=1):
        image = PIL.Image.open(path).convert("RGB")
        if not args.quiet:
            print(f"[{idx}/{len(paths)}] {os.path.basename(path)}")

        t0 = time.perf_counter()
        prompt_results, prof = runner.infer_image(
            image,
            score_thr=args.score_thr,
            mask_thr=args.mask_thr,
            process_area_thr=args.process_area_thr,
            process_prefer=args.process_prefer,
            profile=args.profile,
        )
        if temporal_voter is not None:
            prompt_results = temporal_voter.update(prompt_results, order_idx=idx - 1, path=path)
        if args.tune_precision:
            tune_records.append({
                "image": image.copy(),
                "path": path,
                "prompt_results": clone_prompt_results(prompt_results),
            })
        t1 = time.perf_counter()
        row = finalize_usls_prompt_results_row(
            image=image,
            image_path=path,
            prompt_results=prompt_results,
            out_dir=args.out_dir,
            category_min_scores=min_scores,
            merge_nms_iou=args.merge_nms_iou,
            mask_thr=0.5,
            top_k=args.top_k,
            mask_nms_iou=args.mask_nms_iou,
            mask_nms_overlap=args.mask_nms_overlap,
            save_overlay=not args.no_save_overlays,
            quiet=args.quiet,
            cross_category_mask_nms=args.cross_category_mask_nms,
            cross_category_iou=args.cross_category_iou,
            cross_category_overlap=args.cross_category_overlap,
            category_score_scales={
                "transparent": args.transparent_score_scale,
                "metal": args.metal_score_scale,
                "Cardboard": args.cardboard_score_scale,
            },
        )
        t2 = time.perf_counter()
        row["timing_ms"] = {
            "inference": float((t1 - t0) * 1000.0),
            "postprocess_save": float((t2 - t1) * 1000.0),
            "total": float((t2 - t0) * 1000.0),
        }
        if prof:
            row["usls_profile_ms"] = prof
        if temporal_voter is not None:
            row["temporal_vote"] = {
                "enabled": True,
                "dx_px_per_frame": float(args.conveyor_dx_px_per_frame),
                "dy_px_per_frame": float(args.conveyor_dy_px_per_frame),
                "delta_mode": args.temporal_delta_mode,
                "iou_thr": float(args.temporal_iou_thr),
                "vote_thr": float(args.temporal_vote_thr),
                "current_weight": float(args.temporal_current_weight),
                "history_weight": float(args.temporal_history_weight),
                "decay": float(args.temporal_decay),
                "max_age": int(args.temporal_max_age),
                "inject_missed": bool(args.temporal_inject_missed),
                "missed_min_score": float(args.temporal_missed_min_score),
            }
            if motion_init is not None:
                row["temporal_vote"]["auto_motion_init"] = motion_init
        rows.append(row)
        timings.append(row["timing_ms"])

    with open(pred_jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg = {
        key: float(np.mean([t[key] for t in timings]))
        for key in ("inference", "postprocess_save", "total")
    }
    print(f"saved predictions: {pred_jsonl_path}")
    print(
        "avg_time_ms "
        f"infer={avg['inference']:.2f} post={avg['postprocess_save']:.2f} total={avg['total']:.2f}"
    )

    if not args.no_eval and args.annotations and os.path.exists(args.annotations):
        gt_by_file = iitp.load_annotations_jsonl(args.annotations)
        eval_y_min = None if args.no_eval_y_filter else args.crop_y_min
        eval_y_max = None if args.no_eval_y_filter else args.crop_y_max
        metrics = iitp.evaluate_predictions(
            gt_by_file,
            rows,
            iou_thr=args.eval_iou_thr,
            y_min=eval_y_min,
            y_max=eval_y_max,
            y_filter_mode=args.crop_y_mode,
        )
        iitp.print_eval_metrics(metrics)
        eval_path = os.path.join(args.out_dir, "eval_metrics.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"saved eval metrics: {eval_path}")

        if args.tune_precision and tune_records:
            print(
                "[precision tune] searching threshold/conflict settings "
                f"with recall_floor={args.tune_recall_floor:.3f}"
            )
            tune = tune_precision_from_records(tune_records, gt_by_file, args, min_scores)
            tune_path = os.path.join(args.out_dir, "precision_tuning.json")
            with open(tune_path, "w", encoding="utf-8") as f:
                json.dump(tune, f, ensure_ascii=False, indent=2)
            if tune:
                micro = tune["metrics"]["micro"]
                print(
                    "[precision tune] best "
                    f"meets_recall_floor={tune['meets_recall_floor']} "
                    f"precision={float(micro['precision']):.3f} "
                    f"recall={float(micro['recall']):.3f} "
                    f"f1={float(micro['f1']):.3f} "
                    f"min_scores={tune['min_scores']} "
                    f"cross={tune['cross_category_mask_nms']}"
                )
            print(f"saved precision tuning: {tune_path}")


if __name__ == "__main__":
    main()
