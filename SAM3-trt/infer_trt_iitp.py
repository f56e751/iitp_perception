import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
    
try:
    from transformers.models.sam3 import Sam3Processor
except ImportError:  # only the ONNX-export CLI at the bottom needs it
    Sam3Processor = None
from PIL import Image
import numpy as np
import time
import cv2
import torch
import os
import glob
import argparse
import json
import collections
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
STATIC_BATCH = 5
TEXT_SEQ_LEN = 32
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPT_SPECS = (
    ("transparent", "plastic beverage bottle"),
    ("metal", "recyclable metal packaging"),
    ("metal", "crushed can"),
    ("Cardboard", "cardboard box"),
    ("Cardboard", "cardboard package"),
)
PROMPT_FILE_STEMS = tuple(dict.fromkeys(name for name, _prompt in PROMPT_SPECS))
PROMPTS = tuple(prompt for _name, prompt in PROMPT_SPECS)
CONVEYOR_STEM = "conveyor_belt"
PROMPT_COLORS_RGB = {
    "transparent": (0, 210, 255),
    "metal": (255, 80, 40),
    "Cardboard": (70, 190, 80),
    "conveyor_belt": (255, 220, 0),
}
DEFAULT_CATEGORY_MIN_SCORES = {
    "transparent": 0.8957053,
    "metal": 0.0,
    "Cardboard": 0.9043130,
    "conveyor_belt": 0.0,
}

fl_x = 901.994
fl_y = 901.597
cx   = 650.984
cy   = 367.533
W_in = 1280
H_in = 720

K = np.array([
    [fl_x, 0.0,  cx],
    [0.0,  fl_y, cy],
    [0.0,  0.0,  1.0]
], dtype=np.float64)
n_w  = np.array([0, 0, 1], dtype=np.float64)
p0_w = np.array([0.0, 0.0, 0.0], dtype=np.float64)






def sig(x):
 return 1/(1 + np.exp(-np.clip(x, -80, 80)))

def _np_dtype_from_trt(dtype: trt.DataType) -> np.dtype:
    if dtype == trt.DataType.FLOAT:  return np.float32
    if dtype == trt.DataType.HALF:   return np.float16
    if dtype == trt.DataType.INT8:   return np.int8
    if dtype == trt.DataType.INT32:  return np.int32
    if hasattr(trt.DataType, "INT64") and dtype == trt.DataType.INT64: return np.int64
    if dtype == trt.DataType.BOOL:   return np.bool_
    raise TypeError(f"Unsupported TRT dtype: {dtype}")

def load_engine(engine_path: str) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine.")
    return engine

class Sam3TRTRunner:
    """
    - engine/context/stream: 1회 생성
    - input/output device buffers: shape이 바뀌지 않는 한 재사용
    - infer()는 입력 복사 + execute + 출력 복사만 수행
    """
    def __init__(self, engine_path: str):
        self.engine = load_engine(engine_path)

        if not (hasattr(self.engine, "num_io_tensors") and hasattr(self.engine, "get_tensor_name")):
            raise RuntimeError("This runner expects Tensor I/O API (TensorRT 9/10 style).")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        self.stream = cuda.Stream()

        # cache tensor names
        n_io = self.engine.num_io_tensors
        self.io_names = [self.engine.get_tensor_name(i) for i in range(n_io)]
        self.in_names = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out_names = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        # cache dtype expectations
        self.in_dtypes = {n: _np_dtype_from_trt(self.engine.get_tensor_dtype(n)) for n in self.in_names}
        self.out_dtypes = {n: _np_dtype_from_trt(self.engine.get_tensor_dtype(n)) for n in self.out_names}
        self.input_shapes = {n: tuple(self.engine.get_tensor_shape(n)) for n in self.in_names}
        self.static_batch = next((shape[0] for shape in self.input_shapes.values() if len(shape) > 0 and shape[0] > 0), STATIC_BATCH)
        self.text_seq_len = next(
            (
                shape[1]
                for name, shape in self.input_shapes.items()
                if ("input_ids" in name.lower() or "attention_mask" in name.lower()) and len(shape) > 1 and shape[1] > 0
            ),
            TEXT_SEQ_LEN,
        )

        # allocate buffers lazily on first infer (need concrete shapes)
        self._cur_in_shapes = None
        self.d_in = {}
        self.h_out = {}
        self.d_out = {}

        # output name mapping (optional heuristic)
        self.pred_key = None
        self.sem_key = None
        self.logit_key = None
        for n in self.out_names:
            ln = n.lower()
            if self.pred_key is None and "mask" in ln:
                self.pred_key = n
            if self.sem_key is None and ("seg" in ln or "semantic" in ln):
                self.sem_key = n
            if self.logit_key is None and "logit" in ln:
                self.logit_key = n
        # fallback
        if self.pred_key is None and len(self.out_names) > 0:
            self.pred_key = self.out_names[0]
        if self.sem_key is None and len(self.out_names) > 1:
            self.sem_key = self.out_names[1]
        if self.logit_key is None and len(self.out_names) > 2:
            self.logit_key = self.out_names[2]

    def _ensure_buffers(self, inputs: dict[str, np.ndarray]):
        """
        input shape이 바뀌면(context set + output shapes 재계산) 버퍼 재할당
        """
        cur_in_shapes = {name: tuple(arr.shape) for name, arr in inputs.items()}
        if self._cur_in_shapes == cur_in_shapes:
            return

        self._cur_in_shapes = cur_in_shapes

        for name in self.in_names:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT input '{name}'. Provided: {list(inputs.keys())}")
            self.context.set_input_shape(name, cur_in_shapes[name])

        # (re)allocate input device buffers
        self.d_in.clear()
        for name in self.in_names:
            arr = inputs[name]
            nbytes_in = int(np.prod(arr.shape)) * np.dtype(self.in_dtypes[name]).itemsize
            dev = cuda.mem_alloc(nbytes_in)
            self.context.set_tensor_address(name, int(dev))
            self.d_in[name] = dev

        # (re)allocate outputs
        self.h_out.clear()
        self.d_out.clear()

        for name in self.out_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.out_dtypes[name]
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype).reshape(shape)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            self.h_out[name] = host
            self.d_out[name] = dev

    def _copy_selected_outputs(self, score_thr: float):
        cuda.memcpy_dtoh_async(self.h_out[self.logit_key], self.d_out[self.logit_key], self.stream)
        if self.sem_key is not None:
            cuda.memcpy_dtoh_async(self.h_out[self.sem_key], self.d_out[self.sem_key], self.stream)
        self.stream.synchronize()

        logits = self.h_out[self.logit_key]
        keep_flat = np.flatnonzero(sig(logits.reshape(-1)) > score_thr)
        pred_shape = self.h_out[self.pred_key].shape
        mask_h, mask_w = pred_shape[-2], pred_shape[-1]
        mask_itemsize = np.dtype(self.out_dtypes[self.pred_key]).itemsize
        mask_nbytes = mask_h * mask_w * mask_itemsize
        flat_pred = self.h_out[self.pred_key].reshape(-1, mask_h, mask_w)
        pred_dev = int(self.d_out[self.pred_key])

        if keep_flat.size > flat_pred.shape[0] * 0.75:
            cuda.memcpy_dtoh_async(self.h_out[self.pred_key], self.d_out[self.pred_key], self.stream)
            copied_masks = int(flat_pred.shape[0])
        else:
            for idx in keep_flat:
                offset = int(idx) * mask_nbytes
                cuda.memcpy_dtoh_async(flat_pred[int(idx)], pred_dev + offset, self.stream)
            copied_masks = int(keep_flat.size)
        return copied_masks

    def infer(
        self,
        inputs: dict[str, np.ndarray],
        profile: bool = True,
        score_thr: float = 0.5,
        selective_mask_d2h: bool = True,
    ):
        """
        inputs: dict of TensorRT inputs such as pixel_values/input_ids/attention_mask
        returns:
          - (pred, sem, logit) if profile=False
          - (pred, sem, logit, prof_dict) if profile=True
        """
        for name in self.in_names:
            if inputs[name].dtype != self.in_dtypes[name]:
                raise TypeError(f"Input dtype mismatch for {name}: expect {self.in_dtypes[name]}, got {inputs[name].dtype}")

        self._ensure_buffers(inputs)

        if not profile:
            # H2D
            for name in self.in_names:
                cuda.memcpy_htod_async(self.d_in[name], inputs[name], self.stream)

            # execute
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execution failed.")

            # D2H (copy only masks whose logits pass score_thr; this avoids a large full-mask transfer)
            if selective_mask_d2h:
                self._copy_selected_outputs(score_thr)
            else:
                cuda.memcpy_dtoh_async(self.h_out[self.pred_key], self.d_out[self.pred_key], self.stream)
                cuda.memcpy_dtoh_async(self.h_out[self.sem_key], self.d_out[self.sem_key], self.stream)
                cuda.memcpy_dtoh_async(self.h_out[self.logit_key], self.d_out[self.logit_key], self.stream)

            self.stream.synchronize()
            return self.h_out[self.pred_key], self.h_out[self.sem_key], self.h_out[self.logit_key]

        # ---------------------------
        # Profiling path (CUDA events)
        # ---------------------------
        # Create events (you can cache these in __init__ if you want)
        e0 = cuda.Event()
        e1 = cuda.Event()
        e2 = cuda.Event()
        e3 = cuda.Event()
        e4 = cuda.Event()

        t_cpu0 = time.perf_counter()

        # mark start
        e0.record(self.stream)

        # H2D
        for name in self.in_names:
            cuda.memcpy_htod_async(self.d_in[name], inputs[name], self.stream)
        e1.record(self.stream)

        # execute
        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        e2.record(self.stream)

        # D2H
        if selective_mask_d2h:
            copied_masks = self._copy_selected_outputs(score_thr)
        else:
            cuda.memcpy_dtoh_async(self.h_out[self.pred_key], self.d_out[self.pred_key], self.stream)
            cuda.memcpy_dtoh_async(self.h_out[self.sem_key], self.d_out[self.sem_key], self.stream)
            cuda.memcpy_dtoh_async(self.h_out[self.logit_key], self.d_out[self.logit_key], self.stream)
            copied_masks = int(np.prod(self.h_out[self.pred_key].shape[:2]))
        e3.record(self.stream)

        # finish
        e4.record(self.stream)
        e4.synchronize()

        t_cpu1 = time.perf_counter()

        # elapsed_time returns milliseconds (float)
        ms_total = e0.time_till(e4)
        ms_h2d   = e0.time_till(e1)
        ms_exec  = e1.time_till(e2)
        ms_d2h   = e2.time_till(e3)
        ms_tail  = e3.time_till(e4)  # usually tiny, but shows sync/overhead

        prof = {
            "ms_total_cuda": float(ms_total),
            "ms_h2d": float(ms_h2d),
            "ms_exec": float(ms_exec),
            "ms_d2h": float(ms_d2h),
            "ms_tail": float(ms_tail),
            "ms_total_cpu": float((t_cpu1 - t_cpu0) * 1000.0),
            "copied_masks": copied_masks,
            "shapes": {k: tuple(v.shape) for k, v in inputs.items()},
            "dtypes": {k: str(v.dtype) for k, v in inputs.items()},
        }

        return self.h_out[self.pred_key], self.h_out[self.sem_key], self.h_out[self.logit_key], prof
    

def to_uint8_rgb(pil_img):
    arr = np.array(pil_img)  # HWC RGB uint8
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

def make_color_lut(n=256, seed=0):
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    lut[0] = np.array([0, 0, 0], np.uint8)
    return lut

def bbox_from_mask(mask_bool: np.ndarray):
    # mask_bool: (H,W) bool
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def box_iou(a, b):
    # a,b: [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1 + 1.0)
    ih = max(0.0, inter_y2 - inter_y1 + 1.0)
    inter = iw * ih
    area_a = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    area_b = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)

def mask_iou(a, b):
    # a,b: bool (H,W)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-6)

def warp_mask(mask_bool: np.ndarray, H: np.ndarray, out_h: int, out_w: int):
    # mask_bool: (H,W) bool
    src = (mask_bool.astype(np.uint8) * 255)
    warped = cv2.warpPerspective(src, H, (out_w, out_h), flags=cv2.INTER_NEAREST)
    return warped > 127


def process_masks_by_size(masks_bool: np.ndarray, threshold: int = 1000, prefer: str = "smaller"):
    if masks_bool.size == 0:
        return np.empty((0, *masks_bool.shape[1:]), dtype=bool), np.empty((0,), dtype=np.int64)

    masks_bool = masks_bool.astype(bool).copy()
    n_masks, height, width = masks_bool.shape

    areas = masks_bool.reshape(n_masks, -1).sum(axis=1)
    order = np.argsort(areas)
    if prefer == "larger":
        order = order[::-1]

    occupied = np.zeros((height, width), dtype=bool)
    kept = []
    kept_indices = []
    for idx in order:
        cur = masks_bool[idx] & (~occupied)
        if cur.sum() >= threshold:
            kept.append(cur)
            kept_indices.append(idx)
            occupied |= cur

    if kept:
        return np.stack(kept, axis=0), np.asarray(kept_indices, dtype=np.int64)
    return np.empty((0, height, width), dtype=bool), np.empty((0,), dtype=np.int64)



def overlay_masks(
    image_rgb_u8: np.ndarray,
    masks_prob: np.ndarray,
    scores: np.ndarray | None = None,
    alpha: float = 0.45,
    thr: float = 0.5,
    draw_contour: bool = True,
    thickness: int = 2,
    top_k: int | None = None,
    seed: int = 0,
):
    """
    image_rgb_u8: (H,W,3) uint8
    masks_prob:   (N,H,W) float in [0,1]
    scores:       (N,) float optional
    returns:      overlay RGB uint8
    """
    H, W, _ = image_rgb_u8.shape
    N = masks_prob.shape[0]
    assert masks_prob.shape[1:] == (H, W)

    if scores is None:
        scores = np.ones((N,), dtype=np.float32)
    else:
        scores = scores.astype(np.float32).reshape(-1)

    order = np.argsort(-scores)
    if top_k is not None:
        order = order[:top_k]

    lut = make_color_lut(max(N+1, 256), seed=seed)

    out = image_rgb_u8.astype(np.float32).copy()

    for idx_rank, i in enumerate(order, start=1):
        m = masks_prob[i] >= thr
        if not np.any(m):
            continue

        color = lut[idx_rank % lut.shape[0]].astype(np.float32)  # RGB
        # alpha blend only where mask==1
        out[m] = out[m] * (1 - alpha) + color * alpha

        if draw_contour:
            # cv2 expects uint8 mask
            m_u8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # drawContours expects BGR, so convert color
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            out_bgr = out[..., ::-1].astype(np.uint8)
            cv2.drawContours(out_bgr, contours, -1, bgr, thickness=thickness)
            out = out_bgr[..., ::-1].astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def build_static_batch_inputs(
    processor,
    images,
    static_batch: int = STATIC_BATCH,
    text_seq_len: int = TEXT_SEQ_LEN,
    prompts: tuple[str, ...] = PROMPTS,
    include_text: bool = True,
):
    if len(images) == 0:
        raise ValueError("At least one image is required.")
    if len(images) != 1:
        raise ValueError(f"IITP layout expects exactly 1 image per TRT batch, got {len(images)}")
    if len(prompts) > static_batch:
        raise ValueError(f"Received {len(prompts)} prompts, but engine static batch is {static_batch}")

    padded_images = []
    padded_prompts = []
    dummy_image = Image.new("RGB", images[0].size, (0, 0, 0))

    for prompt in prompts:
        padded_images.append(images[0])
        padded_prompts.append(prompt)

    while len(padded_images) < static_batch:
        padded_images.append(dummy_image)
        padded_prompts.append(prompts[-1])

    padded_images = padded_images[:static_batch]
    padded_prompts = padded_prompts[:static_batch]

    image_inputs = processor.image_processor(images=padded_images, return_tensors="pt")
    result = {
        "pixel_values": image_inputs["pixel_values"].cpu().numpy().astype(np.float32),
        "valid_image_count": len(images),
        "prompts": padded_prompts,
    }
    if include_text:
        text_inputs = processor.tokenizer(
            padded_prompts,
            padding="max_length",
            truncation=True,
            max_length=text_seq_len,
            return_tensors="pt",
        )
        result["input_ids"] = text_inputs["input_ids"].cpu().numpy()
        result["attention_mask"] = text_inputs["attention_mask"].cpu().numpy()
    return result


def select_outputs_for_image(pred_masks_logits: np.ndarray, logit: np.ndarray, image_idx: int, prompt_idx: int):
    if image_idx != 0:
        raise ValueError("IITP layout runs one image per TRT batch.")
    slot = prompt_idx
    return pred_masks_logits[slot:slot + 1], logit[slot:slot + 1]


def decode_slot_masks(
    pred_masks_logits: np.ndarray,
    logit: np.ndarray,
    image_size_hw: tuple[int, int],
    score_thr: float = 0.5,
    mask_thr: float = 0.5,
):
    image_h, image_w = image_size_hw
    score_flat = logit.reshape(-1)
    keep_flat = sig(score_flat) > score_thr

    if keep_flat.sum() == 0:
        empty_masks = np.empty((0, image_h, image_w), dtype=np.float32)
        empty_scores = np.empty((0,), dtype=np.float32)
        return empty_masks, empty_scores

    masks_logits = pred_masks_logits.reshape(-1, pred_masks_logits.shape[-2], pred_masks_logits.shape[-1])[keep_flat]
    scores_kept = sig(score_flat[keep_flat]).astype(np.float32)

    masks_prob = np.empty((masks_logits.shape[0], image_h, image_w), dtype=np.float32)
    for i, mask_logit in enumerate(masks_logits):
        resized = cv2.resize(mask_logit.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        masks_prob[i] = sig(resized).astype(np.float32)
    return masks_prob, scores_kept


def decode_processed_slot(
    image: Image.Image,
    pred_masks_logits: np.ndarray,
    logit: np.ndarray,
    score_thr: float = 0.5,
    mask_thr: float = 0.5,
    process_area_thr: int = 1000,
    process_prefer: str = "smaller",
):
    masks_prob, scores_kept = decode_slot_masks(
        pred_masks_logits,
        logit,
        (image.height, image.width),
        score_thr=score_thr,
        mask_thr=mask_thr,
    )
    raw_count = masks_prob.shape[0]
    processed_masks_bool, processed_indices = process_masks_by_size(
        masks_prob >= mask_thr,
        threshold=process_area_thr,
        prefer=process_prefer,
    )
    masks_prob = processed_masks_bool.astype(np.float32)
    scores_kept = scores_kept[processed_indices] if processed_indices.size > 0 else scores_kept[:0]
    return masks_prob, scores_kept, raw_count


def dilate_mask(mask_bool: np.ndarray, kernel_size: int):
    if kernel_size <= 1 or not np.any(mask_bool):
        return mask_bool
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0


def suppress_masks(masks_prob: np.ndarray, scores: np.ndarray, suppress_mask: np.ndarray, min_area: int):
    if masks_prob.size == 0 or suppress_mask is None or not np.any(suppress_mask):
        return masks_prob, scores

    masks_bool = (masks_prob > 0.5) & (~suppress_mask[None, :, :])
    areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
    keep = areas >= min_area
    if keep.sum() == 0:
        return np.empty((0, *masks_prob.shape[1:]), dtype=np.float32), scores[:0]
    return masks_bool[keep].astype(np.float32), scores[keep]


def masks_conflict(a: np.ndarray, b: np.ndarray, iou_thr: float, overlap_thr: float):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return False
    area_a = a.sum()
    area_b = b.sum()
    union = area_a + area_b - inter
    iou = inter / (union + 1e-6)
    overlap = inter / (min(area_a, area_b) + 1e-6)
    return iou >= iou_thr or overlap >= overlap_thr


def resolve_prompt_masks_by_score(
    prompt_results: dict,
    mask_thr: float,
    iou_thr: float = 0.3,
    overlap_thr: float = 0.6,
    include_conveyor: bool = False,
):
    candidates = []
    target_prompts = [
        name for name in prompt_results.keys()
        if include_conveyor or name != CONVEYOR_STEM
    ]
    for prompt_name in target_prompts:
        masks_prob = prompt_results[prompt_name]["masks"]
        scores = prompt_results[prompt_name]["scores"]
        for mask_idx, score in enumerate(scores):
            mask_bool = masks_prob[mask_idx] >= mask_thr
            if np.any(mask_bool):
                candidates.append({
                    "prompt_name": prompt_name,
                    "score": float(score),
                    "mask": mask_bool,
                })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    kept = []
    removed = collections.Counter()
    for candidate in candidates:
        if any(masks_conflict(candidate["mask"], item["mask"], iou_thr, overlap_thr) for item in kept):
            removed[candidate["prompt_name"]] += 1
            continue
        kept.append(candidate)

    height = None
    width = None
    for result in prompt_results.values():
        if result["masks"].size > 0:
            height, width = result["masks"].shape[1:]
            break
    if height is None or width is None:
        return removed

    grouped_masks = {name: [] for name in target_prompts}
    grouped_scores = {name: [] for name in target_prompts}
    for item in kept:
        grouped_masks[item["prompt_name"]].append(item["mask"].astype(np.float32))
        grouped_scores[item["prompt_name"]].append(item["score"])

    for prompt_name in target_prompts:
        if grouped_masks[prompt_name]:
            prompt_results[prompt_name]["masks"] = np.stack(grouped_masks[prompt_name], axis=0)
            prompt_results[prompt_name]["scores"] = np.asarray(grouped_scores[prompt_name], dtype=np.float32)
        else:
            prompt_results[prompt_name]["masks"] = np.empty((0, height, width), dtype=np.float32)
            prompt_results[prompt_name]["scores"] = np.empty((0,), dtype=np.float32)
        prompt_results[prompt_name]["mask_nms_removed"] = int(removed[prompt_name])

    return removed


def render_overlay_from_masks(
    image: Image.Image,
    masks_prob: np.ndarray,
    scores_kept: np.ndarray,
    mask_thr: float = 0.5,
    top_k: int | None = None,
):

    img_u8 = to_uint8_rgb(image)
    overlay = overlay_masks(
        img_u8,
        masks_prob,
        scores=scores_kept,
        alpha=0.45,
        thr=mask_thr,
        draw_contour=True,
        thickness=2,
        top_k=top_k,
        seed=0,
    )
    return overlay


def render_prompt_color_overlay(
    image: Image.Image,
    prompt_results: dict,
    mask_thr: float = 0.5,
    top_k: int | None = None,
    alpha: float = 0.45,
):
    img_u8 = to_uint8_rgb(image)
    out = img_u8.astype(np.float32)
    prompt_regions = {}

    for prompt_name in prompt_results.keys():
        masks_prob = prompt_results[prompt_name]["masks"]
        scores = prompt_results[prompt_name]["scores"]
        if masks_prob.size == 0 or scores.size == 0:
            continue

        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        masks_bool = masks_prob[order] >= mask_thr
        if masks_bool.size == 0:
            continue

        prompt_regions[prompt_name] = masks_bool.any(axis=0)

    for prompt_name in prompt_results.keys():
        region = prompt_regions.get(prompt_name)
        if region is None:
            continue
        if not np.any(region):
            continue
        color = np.array(PROMPT_COLORS_RGB[prompt_name], dtype=np.float32)
        out[region] = out[region] * (1 - alpha) + color * alpha

    out_bgr = out[..., ::-1].astype(np.uint8)
    for prompt_name in prompt_results.keys():
        region = prompt_regions.get(prompt_name)
        if region is None:
            continue
        if not np.any(region):
            continue
        color = np.array(PROMPT_COLORS_RGB[prompt_name], dtype=np.float32)
        contour_color = (int(color[2]), int(color[1]), int(color[0]))
        m_u8 = (region.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_bgr, contours, -1, contour_color, thickness=2)
    out = out_bgr[..., ::-1].astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def save_image_prompt_overlays(
    images: list[Image.Image],
    image_paths: list[str],
    pred_masks_logits: np.ndarray,
    logit: np.ndarray,
    out_dir: str,
    valid_image_count: int,
    score_thr: float = 0.5,
    mask_thr: float = 0.5,
    top_k: int | None = None,
    process_area_thr: int = 1000,
    process_prefer: str = "smaller",
    suppress_conveyor: bool = True,
    conveyor_dilate: int = 7,
    save_per_prompt: bool = False,
    prompts: tuple[str, ...] = PROMPTS,
    mask_nms: bool = True,
    mask_nms_iou: float = 0.3,
    mask_nms_overlap: float = 0.6,
    save_overlay: bool = True,
    quiet: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    prediction_rows = []

    for image_idx in range(valid_image_count):
        if image_idx != 0:
            raise ValueError("IITP layout runs one image per TRT batch.")

        stem = os.path.splitext(os.path.basename(image_paths[image_idx]))[0]
        prompt_results = {}

        for prompt_idx, prompt_name in enumerate(PROMPT_FILE_STEMS):
            slot_masks, slot_logit = select_outputs_for_image(pred_masks_logits, logit, image_idx, prompt_idx)
            masks_prob, scores_kept, raw_count = decode_processed_slot(
                images[image_idx],
                slot_masks,
                slot_logit,
                score_thr=score_thr,
                mask_thr=mask_thr,
                process_area_thr=process_area_thr,
                process_prefer=process_prefer,
            )
            prompt_results[prompt_name] = {
                "masks": masks_prob,
                "scores": scores_kept,
                "raw_count": raw_count,
            }

        conveyor_masks = prompt_results[CONVEYOR_STEM]["masks"]
        if conveyor_masks.size > 0:
            conveyor_union = (conveyor_masks >= mask_thr).any(axis=0)
            conveyor_union = dilate_mask(conveyor_union, conveyor_dilate)
        else:
            conveyor_union = np.zeros((images[image_idx].height, images[image_idx].width), dtype=bool)

        if suppress_conveyor:
            for prompt_name in PROMPT_FILE_STEMS:
                if prompt_name == CONVEYOR_STEM:
                    continue
                masks_prob, scores_kept = suppress_masks(
                    prompt_results[prompt_name]["masks"],
                    prompt_results[prompt_name]["scores"],
                    conveyor_union,
                    min_area=process_area_thr,
                )
                prompt_results[prompt_name]["masks"] = masks_prob
                prompt_results[prompt_name]["scores"] = scores_kept

        if mask_nms:
            resolve_prompt_masks_by_score(
                prompt_results,
                mask_thr=mask_thr,
                iou_thr=mask_nms_iou,
                overlap_thr=mask_nms_overlap,
                include_conveyor=False,
            )

        row = {
            "filename": os.path.basename(image_paths[image_idx]),
            "height": images[image_idx].height,
            "width": images[image_idx].width,
            "conveyor_suppression": bool(suppress_conveyor),
            "mask_nms": bool(mask_nms),
            "mask_nms_iou": float(mask_nms_iou),
            "mask_nms_overlap": float(mask_nms_overlap),
            "prompts": {},
            "instances": [],
        }

        for prompt_name in PROMPT_FILE_STEMS:
            masks_prob = prompt_results[prompt_name]["masks"]
            scores_kept = prompt_results[prompt_name]["scores"]
            raw_count = prompt_results[prompt_name]["raw_count"]

            if scores_kept.size > 0:
                order = np.argsort(-scores_kept)
                if top_k is not None:
                    order = order[:top_k]
            else:
                order = np.empty((0,), dtype=np.int64)

            row["prompts"][prompt_name] = {
                "prompt": prompts[PROMPT_FILE_STEMS.index(prompt_name)],
                "color_rgb": list(PROMPT_COLORS_RGB[prompt_name]),
                "raw_count": int(raw_count),
                "processed_count": int(scores_kept.size),
                "mask_nms_removed": int(prompt_results[prompt_name].get("mask_nms_removed", 0)),
                "top_score": float(scores_kept[order[0]]) if order.size > 0 else 0.0,
            }

            if save_per_prompt:
                overlay = render_overlay_from_masks(
                    images[image_idx],
                    masks_prob,
                    scores_kept,
                    mask_thr=mask_thr,
                    top_k=top_k,
                )
                out_path = os.path.join(out_dir, f"{stem}_{prompt_name}.png")
                if save_overlay:
                    cv2.imwrite(out_path, overlay[..., ::-1])

                if order.size > 0:
                    top_mask_u8 = ((masks_prob[order[0]] >= mask_thr).astype(np.uint8) * 255)
                    union_mask_u8 = ((masks_prob[order] >= mask_thr).any(axis=0).astype(np.uint8) * 255)
                else:
                    top_mask_u8 = np.zeros((images[image_idx].height, images[image_idx].width), dtype=np.uint8)
                    union_mask_u8 = top_mask_u8

                mask_path = os.path.join(out_dir, f"{stem}_{prompt_name}_mask.png")
                union_mask_path = os.path.join(out_dir, f"{stem}_{prompt_name}_union_mask.png")
                if save_overlay:
                    cv2.imwrite(mask_path, top_mask_u8)
                    cv2.imwrite(union_mask_path, union_mask_u8)
                row["prompts"][prompt_name]["mask_path"] = mask_path
                row["prompts"][prompt_name]["union_mask_path"] = union_mask_path
                row["prompts"][prompt_name]["overlay_path"] = out_path

            for mask_idx, score in enumerate(scores_kept):
                box = bbox_from_mask(masks_prob[mask_idx] >= mask_thr)
                if box is None:
                    continue
                row["instances"].append({
                    "category": prompt_name,
                    "score": float(score),
                    "bbox_xyxy": [float(x) for x in box.tolist()],
                })

            if not quiet:
                print(
                    f"{stem} {prompt_name}: raw={raw_count} processed={scores_kept.size} "
                    f"mask_nms_removed={prompt_results[prompt_name].get('mask_nms_removed', 0)} "
                    f"top_score={(float(scores_kept[order[0]]) if order.size > 0 else 0.0):.3f} "
                )

        combined_path = os.path.join(out_dir, f"{stem}_prompts.png")
        if save_overlay:
            combined_overlay = render_prompt_color_overlay(
                images[image_idx],
                prompt_results,
                mask_thr=mask_thr,
                top_k=top_k,
            )
            cv2.imwrite(combined_path, combined_overlay[..., ::-1])
            row["overlay_path"] = combined_path
            if not quiet:
                print(f"{stem}: saved={combined_path}")

        prediction_rows.append(row)

    return prediction_rows


def empty_prompt_result(height: int, width: int):
    return {
        "masks": np.empty((0, height, width), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "raw_count": 0,
        "mask_nms_removed": 0,
        "prompts": [],
    }


def append_prompt_result(dst: dict, src: dict):
    if src["masks"].size > 0:
        if dst["masks"].size > 0:
            dst["masks"] = np.concatenate([dst["masks"], src["masks"]], axis=0)
            dst["scores"] = np.concatenate([dst["scores"], src["scores"]], axis=0)
        else:
            dst["masks"] = src["masks"].copy()
            dst["scores"] = src["scores"].copy()
    dst["raw_count"] += int(src.get("raw_count", 0))
    dst["mask_nms_removed"] += int(src.get("mask_nms_removed", 0))
    for prompt in src.get("prompts", []):
        if prompt not in dst["prompts"]:
            dst["prompts"].append(prompt)


def filter_prompt_result_by_score(result: dict, min_score: float):
    if min_score <= 0.0 or result["scores"].size == 0:
        return
    keep = result["scores"] >= min_score
    result["masks"] = result["masks"][keep] if keep.any() else result["masks"][:0]
    result["scores"] = result["scores"][keep] if keep.any() else result["scores"][:0]


def nms_prompt_result_by_bbox(result: dict, mask_thr: float, iou_thr: float):
    if result["scores"].size <= 1:
        return
    order = np.argsort(-result["scores"])
    kept = []
    kept_boxes = []
    for idx in order:
        box = bbox_from_mask(result["masks"][idx] >= mask_thr)
        if box is None:
            continue
        box_list = [float(x) for x in box.tolist()]
        if all(box_iou(box_list, prev_box) < iou_thr for prev_box in kept_boxes):
            kept.append(idx)
            kept_boxes.append(box_list)
    if kept:
        kept = np.asarray(kept, dtype=np.int64)
        result["masks"] = result["masks"][kept]
        result["scores"] = result["scores"][kept]
    else:
        result["masks"] = result["masks"][:0]
        result["scores"] = result["scores"][:0]


def decode_prompt_group_results(
    image: Image.Image,
    pred_masks_logits: np.ndarray,
    logit: np.ndarray,
    prompt_specs: tuple[tuple[str, str], ...],
    score_thr: float,
    mask_thr: float,
    process_area_thr: int,
    process_prefer: str,
    suppress_conveyor: bool,
    conveyor_dilate: int,
    mask_nms: bool,
    mask_nms_iou: float,
    mask_nms_overlap: float,
):
    prompt_results = {}
    for prompt_idx, (prompt_name, prompt_text) in enumerate(prompt_specs):
        slot_masks, slot_logit = select_outputs_for_image(pred_masks_logits, logit, 0, prompt_idx)
        masks_prob, scores_kept, raw_count = decode_processed_slot(
            image,
            slot_masks,
            slot_logit,
            score_thr=score_thr,
            mask_thr=mask_thr,
            process_area_thr=process_area_thr,
            process_prefer=process_prefer,
        )
        if prompt_name not in prompt_results:
            prompt_results[prompt_name] = empty_prompt_result(image.height, image.width)
        append_prompt_result(
            prompt_results[prompt_name],
            {
                "masks": masks_prob,
                "scores": scores_kept,
                "raw_count": raw_count,
                "mask_nms_removed": 0,
                "prompts": [prompt_text],
            },
        )

    if CONVEYOR_STEM in prompt_results and prompt_results[CONVEYOR_STEM]["masks"].size > 0:
        conveyor_union = (prompt_results[CONVEYOR_STEM]["masks"] >= mask_thr).any(axis=0)
        conveyor_union = dilate_mask(conveyor_union, conveyor_dilate)
    else:
        conveyor_union = np.zeros((image.height, image.width), dtype=bool)

    if suppress_conveyor:
        for prompt_name, result in prompt_results.items():
            if prompt_name == CONVEYOR_STEM:
                continue
            masks_prob, scores_kept = suppress_masks(
                result["masks"],
                result["scores"],
                conveyor_union,
                min_area=process_area_thr,
            )
            result["masks"] = masks_prob
            result["scores"] = scores_kept

    if mask_nms:
        resolve_prompt_masks_by_score(
            prompt_results,
            mask_thr=mask_thr,
            iou_thr=mask_nms_iou,
            overlap_thr=mask_nms_overlap,
            include_conveyor=False,
        )

    return prompt_results


def finalize_prompt_results_row(
    image: Image.Image,
    image_path: str,
    prompt_results: dict,
    out_dir: str,
    category_min_scores: dict,
    merge_nms_iou: float,
    mask_thr: float,
    top_k: int | None,
    suppress_conveyor: bool,
    mask_nms: bool,
    mask_nms_iou: float,
    mask_nms_overlap: float,
    save_overlay: bool,
    quiet: bool,
):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for prompt_name, min_score in category_min_scores.items():
        if prompt_name in prompt_results:
            filter_prompt_result_by_score(prompt_results[prompt_name], min_score)
    for prompt_name, result in prompt_results.items():
        if prompt_name == CONVEYOR_STEM:
            continue
        nms_prompt_result_by_bbox(result, mask_thr=mask_thr, iou_thr=merge_nms_iou)

    row = {
        "filename": os.path.basename(image_path),
        "height": image.height,
        "width": image.width,
        "conveyor_suppression": bool(suppress_conveyor),
        "mask_nms": bool(mask_nms),
        "mask_nms_iou": float(mask_nms_iou),
        "mask_nms_overlap": float(mask_nms_overlap),
        "merge_nms_iou": float(merge_nms_iou),
        "category_min_scores": {k: float(v) for k, v in category_min_scores.items()},
        "prompts": {},
        "instances": [],
    }

    for prompt_name, result in prompt_results.items():
        scores_kept = result["scores"]
        masks_prob = result["masks"]
        order = np.argsort(-scores_kept) if scores_kept.size > 0 else np.empty((0,), dtype=np.int64)
        if top_k is not None:
            order = order[:top_k]
        row["prompts"][prompt_name] = {
            "prompt": result.get("prompts", []),
            "color_rgb": list(PROMPT_COLORS_RGB[prompt_name]),
            "raw_count": int(result.get("raw_count", 0)),
            "processed_count": int(scores_kept.size),
            "mask_nms_removed": int(result.get("mask_nms_removed", 0)),
            "top_score": float(scores_kept[order[0]]) if order.size > 0 else 0.0,
        }
        for mask_idx, score in enumerate(scores_kept):
            box = bbox_from_mask(masks_prob[mask_idx] >= mask_thr)
            if box is None:
                continue
            row["instances"].append({
                "category": prompt_name,
                "score": float(score),
                "bbox_xyxy": [float(x) for x in box.tolist()],
            })
        if not quiet:
            print(
                f"{stem} {prompt_name}: raw={result.get('raw_count', 0)} "
                f"processed={scores_kept.size} top_score="
                f"{(float(scores_kept[order[0]]) if order.size > 0 else 0.0):.3f}"
            )

    combined_path = os.path.join(out_dir, f"{stem}_prompts.png")
    if save_overlay:
        combined_overlay = render_prompt_color_overlay(
            image,
            prompt_results,
            mask_thr=mask_thr,
            top_k=top_k,
        )
        cv2.imwrite(combined_path, combined_overlay[..., ::-1])
        row["overlay_path"] = combined_path
        if not quiet:
            print(f"{stem}: saved={combined_path}")

    return row


def collect_image_paths(image_dir: str, limit: int | None = None):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(paths)
    if limit is not None:
        paths = paths[:limit]
    return paths


def normalize_eval_category(category: str):
    value = str(category).strip().lower()
    aliases = {
        "transparent": "transparent",
        "clear": "transparent",
        "plastic": "transparent",
        "pet": "transparent",
        "metal": "metal",
        "can": "metal",
        "drink can": "metal",
        "aluminum can": "metal",
        "aluminum drink can": "metal",
        "cardboard": "cardboard",
        "cardboard box": "cardboard",
    }
    return aliases.get(value)


def load_annotations_jsonl(path: str):
    gt_by_file = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            filename = row["filename"]
            instances = []
            for inst in row.get("detection", {}).get("instances", []):
                category = normalize_eval_category(inst.get("category", ""))
                if category is None:
                    continue
                instances.append({
                    "category": category,
                    "bbox": [float(x) for x in inst["bbox"]],
                })
            gt_by_file[filename] = instances
    return gt_by_file


def bbox_passes_y_filter(bbox, y_min: float | None, y_max: float | None, mode: str = "center"):
    if y_min is None and y_max is None:
        return True
    _x1, y1, _x2, y2 = [float(v) for v in bbox]
    if y_min is None:
        y_min = -float("inf")
    if y_max is None:
        y_max = float("inf")
    if mode == "center":
        cy = (y1 + y2) * 0.5
        return y_min <= cy <= y_max
    if mode == "inside":
        return y1 >= y_min and y2 <= y_max
    if mode == "overlap":
        return y2 >= y_min and y1 <= y_max
    raise ValueError(f"Unknown y filter mode: {mode}")


def write_cropped_annotations_jsonl(src_path: str, dst_path: str, y_min: float, y_max: float, mode: str = "center"):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    total = collections.Counter()
    kept = collections.Counter()
    dropped = collections.Counter()
    rows = 0
    rows_with_instances = 0

    with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            instances = row.get("detection", {}).get("instances", [])
            new_instances = []
            for inst in instances:
                category = inst.get("category", "")
                total[category] += 1
                if bbox_passes_y_filter(inst["bbox"], y_min, y_max, mode=mode):
                    new_instances.append(inst)
                    kept[category] += 1
                else:
                    dropped[category] += 1
            row.setdefault("detection", {})["instances"] = new_instances
            if new_instances:
                rows_with_instances += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "src": src_path,
        "dst": dst_path,
        "mode": mode,
        "y_min": float(y_min),
        "y_max": float(y_max),
        "rows": rows,
        "rows_with_instances": rows_with_instances,
        "total": dict(total),
        "kept": dict(kept),
        "dropped": dict(dropped),
    }
    return summary


def average_precision(tp_flags: list[int], fp_flags: list[int], total_gt: int):
    if total_gt <= 0 or not tp_flags:
        return 0.0
    tp_cum = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def evaluate_predictions(
    gt_by_file: dict,
    pred_rows: list[dict],
    iou_thr: float = 0.5,
    y_min: float | None = None,
    y_max: float | None = None,
    y_filter_mode: str = "center",
):
    categories = ("transparent", "metal", "cardboard")
    pred_by_cat = {cat: [] for cat in categories}
    gt_by_cat_file = {cat: collections.defaultdict(list) for cat in categories}

    eval_files = {row["filename"] for row in pred_rows}
    for filename in eval_files:
        for inst in gt_by_file.get(filename, []):
            cat = normalize_eval_category(inst["category"])
            if not bbox_passes_y_filter(inst["bbox"], y_min, y_max, mode=y_filter_mode):
                continue
            if cat in gt_by_cat_file:
                gt_by_cat_file[cat][filename].append(inst["bbox"])

    for row in pred_rows:
        filename = row["filename"]
        for inst in row.get("instances", []):
            cat = normalize_eval_category(inst.get("category", ""))
            if cat not in pred_by_cat:
                continue
            if not bbox_passes_y_filter(inst["bbox_xyxy"], y_min, y_max, mode=y_filter_mode):
                continue
            pred_by_cat[cat].append({
                "filename": filename,
                "score": float(inst.get("score", 0.0)),
                "bbox": [float(x) for x in inst["bbox_xyxy"]],
            })

    metrics = {
        "iou_threshold": float(iou_thr),
        "y_filter": {
            "enabled": y_min is not None or y_max is not None,
            "mode": y_filter_mode,
            "y_min": y_min,
            "y_max": y_max,
        },
        "categories": {},
        "micro": {},
    }
    micro_tp = micro_fp = micro_fn = 0
    matched_ious_all = []

    for cat in categories:
        total_gt = sum(len(v) for v in gt_by_cat_file[cat].values())
        preds = sorted(pred_by_cat[cat], key=lambda x: x["score"], reverse=True)
        matched = {filename: set() for filename in gt_by_cat_file[cat].keys()}
        tp_flags = []
        fp_flags = []
        matched_ious = []

        for pred in preds:
            filename = pred["filename"]
            gt_boxes = gt_by_cat_file[cat].get(filename, [])
            best_iou = 0.0
            best_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched.setdefault(filename, set()):
                    continue
                iou = box_iou(pred["bbox"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            if best_iou >= iou_thr and best_idx >= 0:
                matched[filename].add(best_idx)
                tp_flags.append(1)
                fp_flags.append(0)
                matched_ious.append(best_iou)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp = int(sum(tp_flags))
        fp = int(sum(fp_flags))
        fn = int(total_gt - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ap = average_precision(tp_flags, fp_flags, total_gt)
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        metrics["categories"][cat] = {
            "gt": int(total_gt),
            "pred": int(len(preds)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ap": float(ap),
            "mean_matched_iou": mean_iou,
        }
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        matched_ious_all.extend(matched_ious)

    micro_precision = micro_tp / max(micro_tp + micro_fp, 1)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
    metrics["micro"] = {
        "tp": int(micro_tp),
        "fp": int(micro_fp),
        "fn": int(micro_fn),
        "precision": float(micro_precision),
        "recall": float(micro_recall),
        "f1": float(micro_f1),
        "mean_matched_iou": float(np.mean(matched_ious_all)) if matched_ious_all else 0.0,
    }
    metrics["macro"] = {
        "precision": float(np.mean([m["precision"] for m in metrics["categories"].values()])),
        "recall": float(np.mean([m["recall"] for m in metrics["categories"].values()])),
        "f1": float(np.mean([m["f1"] for m in metrics["categories"].values()])),
        "ap": float(np.mean([m["ap"] for m in metrics["categories"].values()])),
    }
    return metrics


def print_eval_metrics(metrics: dict):
    print(f"eval bbox IoU@{metrics['iou_threshold']:.2f}")
    y_filter = metrics.get("y_filter", {})
    if y_filter.get("enabled"):
        print(
            f"eval y-filter mode={y_filter['mode']} "
            f"y_min={y_filter['y_min']} y_max={y_filter['y_max']}"
        )
    print("category       gt  pred  tp  fp  fn   prec    rec     f1     ap    mIoU")
    for cat, m in metrics["categories"].items():
        print(
            f"{cat:<12} {m['gt']:>4} {m['pred']:>5} {m['tp']:>3} {m['fp']:>3} {m['fn']:>3} "
            f"{m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  {m['ap']:.3f}  {m['mean_matched_iou']:.3f}"
        )
    micro = metrics["micro"]
    macro = metrics["macro"]
    print(
        f"micro        tp={micro['tp']} fp={micro['fp']} fn={micro['fn']} "
        f"prec={micro['precision']:.3f} rec={micro['recall']:.3f} "
        f"f1={micro['f1']:.3f} mIoU={micro['mean_matched_iou']:.3f}"
    )
    print(
        f"macro        prec={macro['precision']:.3f} rec={macro['recall']:.3f} "
        f"f1={macro['f1']:.3f} mAP={macro['ap']:.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 TensorRT on IITP conveyor validation images.")
    parser.add_argument("--image-dir", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "images"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "output", "overlay_outputs_iitp"))
    parser.add_argument("--weights-dir", default=os.path.join(ROOT, "weights", "sam3", "sam3", "assets", "weights"))
    parser.add_argument("--engine-path", default=os.path.join(ROOT, "weights", "sam3", "sam3_f16_mixed_b5.plan"))
    parser.add_argument("--annotations", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "annotations_crop.jsonl"))
    parser.add_argument("--raw-annotations", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "annotations.jsonl"))
    parser.add_argument("--annotation-crop-output", default=os.path.join(ROOT, "data", "perception_dataset_260526", "val", "annotations_crop.jsonl"))
    parser.add_argument("--make-annotation-crop", action="store_true")
    parser.add_argument("--crop-y-min", type=float, default=40.0)
    parser.add_argument("--crop-y-max", type=float, default=440.0)
    parser.add_argument("--crop-y-mode", choices=("center", "inside", "overlap"), default="center")
    parser.add_argument("--no-eval-y-filter", action="store_true")
    parser.add_argument("--eval-iou-thr", type=float, default=0.5)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--transparent-prompt", default="plastic beverage bottle")
    parser.add_argument("--metal-prompt", default="recyclable metal packaging")
    parser.add_argument("--metal-extra-prompt", default="crushed can")
    parser.add_argument("--metal-cardboard-context-prompt", default="metal can or foil package")
    parser.add_argument("--cardboard-prompt", default="cardboard box")
    parser.add_argument("--cardboard-extra-prompt", default="cardboard package")
    parser.add_argument("--conveyor-prompt", default="conveyor belt")
    parser.add_argument("--use-conveyor-prompt", action="store_true")
    parser.add_argument("--ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transparent-min-score", type=float, default=DEFAULT_CATEGORY_MIN_SCORES["transparent"])
    parser.add_argument("--metal-min-score", type=float, default=DEFAULT_CATEGORY_MIN_SCORES["metal"])
    parser.add_argument("--cardboard-min-score", type=float, default=DEFAULT_CATEGORY_MIN_SCORES["Cardboard"])
    parser.add_argument("--conveyor-min-score", type=float, default=DEFAULT_CATEGORY_MIN_SCORES["conveyor_belt"])
    parser.add_argument("--merge-nms-iou", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--mask-thr", type=float, default=0.5)
    parser.add_argument("--area-thr", type=int, default=1000)
    parser.add_argument("--prefer", choices=("smaller", "larger"), default="smaller")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conveyor-dilate", type=int, default=7)
    parser.add_argument("--no-conveyor-suppress", action="store_true")
    parser.add_argument("--no-mask-nms", action="store_true")
    parser.add_argument("--mask-nms-iou", type=float, default=0.3)
    parser.add_argument("--mask-nms-overlap", type=float, default=0.6)
    parser.add_argument("--save-per-prompt", action="store_true")
    parser.add_argument("--no-save-overlays", action="store_true")
    parser.add_argument("--no-selective-mask-d2h", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def prompts_from_args(args) -> tuple[str, ...]:
    return (
        args.transparent_prompt,
        args.metal_prompt,
        args.cardboard_prompt,
        args.conveyor_prompt,
    )


def category_min_scores_from_args(args) -> dict:
    return {
        "transparent": args.transparent_min_score,
        "metal": args.metal_min_score,
        "Cardboard": args.cardboard_min_score,
        "conveyor_belt": args.conveyor_min_score,
    }


def ensemble_groups_from_args(args, static_batch: int):
    optimized_specs = [
        ("transparent", args.transparent_prompt),
        ("metal", args.metal_prompt),
        ("metal", args.metal_extra_prompt),
        ("Cardboard", args.cardboard_prompt),
        ("Cardboard", args.cardboard_extra_prompt),
    ]
    if args.use_conveyor_prompt:
        optimized_specs.append(("conveyor_belt", args.conveyor_prompt))

    if len(optimized_specs) <= static_batch:
        return ({
            "name": "optimized",
            "prompt_specs": tuple(optimized_specs),
            "use_categories": tuple(dict.fromkeys(name for name, _prompt in optimized_specs)),
        },)

    if not args.use_conveyor_prompt and static_batch >= 4:
        return (
            {
                "name": "main_4prompt",
                "prompt_specs": (
                    ("transparent", args.transparent_prompt),
                    ("metal", args.metal_prompt),
                    ("metal", args.metal_extra_prompt),
                    ("Cardboard", args.cardboard_prompt),
                ),
                "use_categories": ("transparent", "metal", "Cardboard"),
            },
            {
                "name": "cardboard_package",
                "prompt_specs": (("Cardboard", args.cardboard_extra_prompt),),
                "use_categories": ("Cardboard",),
            },
        )

    return (
        {
            "name": "base_recyclable",
            "prompt_specs": (
                ("transparent", args.transparent_prompt),
                ("metal", args.metal_prompt),
                ("Cardboard", args.cardboard_prompt),
                ("conveyor_belt", args.conveyor_prompt),
            ),
            "use_categories": ("transparent", "metal", "conveyor_belt"),
        },
        {
            "name": "metal_crushed",
            "prompt_specs": (
                ("metal", args.metal_extra_prompt),
            ),
            "use_categories": ("metal",),
        },
        {
            "name": "cardboard_package",
            "prompt_specs": (
                ("Cardboard", args.cardboard_extra_prompt),
            ),
            "use_categories": ("Cardboard",),
        },
    )


def main():
    args = parse_args()
    prompts = prompts_from_args(args)
    if args.make_annotation_crop or not os.path.exists(args.annotations):
        summary = write_cropped_annotations_jsonl(
            args.raw_annotations,
            args.annotation_crop_output,
            y_min=args.crop_y_min,
            y_max=args.crop_y_max,
            mode=args.crop_y_mode,
        )
        print("wrote cropped annotations:", args.annotation_crop_output)
        print(
            f"crop summary mode={summary['mode']} y=[{summary['y_min']}, {summary['y_max']}] "
            f"rows={summary['rows']} rows_with_instances={summary['rows_with_instances']}"
        )
        print("kept:", summary["kept"])
        print("dropped:", summary["dropped"])
        if args.make_annotation_crop:
            return

    image_paths = collect_image_paths(args.image_dir, limit=args.limit)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    processor = Sam3Processor.from_pretrained(args.weights_dir)
    runner = Sam3TRTRunner(args.engine_path)
    prompt_groups = ensemble_groups_from_args(args, runner.static_batch) if args.ensemble else ({
        "name": "single",
        "prompt_specs": tuple(PROMPT_SPECS),
        "use_categories": tuple(PROMPT_FILE_STEMS),
    },)
    for group in prompt_groups:
        if runner.static_batch < len(group["prompt_specs"]):
            raise RuntimeError(
                f"Engine batch {runner.static_batch} is smaller than prompt count "
                f"{len(group['prompt_specs'])} for group {group['name']}"
            )

    print("ensemble:", bool(args.ensemble))
    for group in prompt_groups:
        print(
            f"prompt_group {group['name']}: "
            f"{[prompt for _name, prompt in group['prompt_specs']]} "
            f"use={list(group['use_categories'])}"
        )
    print("category_min_scores:", category_min_scores_from_args(args))
    print(f"images: {len(image_paths)} from {args.image_dir}")
    print(f"out_dir: {args.out_dir}")

    pred_jsonl_path = os.path.join(args.out_dir, "predictions.jsonl")
    timing_rows = []
    pred_rows_all = []
    with open(pred_jsonl_path, "w", encoding="utf-8") as f:
        for idx, image_path in enumerate(image_paths, start=1):
            image_t0 = time.perf_counter()
            image = Image.open(image_path).convert("RGB")
            pre_ms = inf_ms = post_ms = 0.0
            prof_acc = collections.Counter()
            merged_results = {
                name: empty_prompt_result(image.height, image.width)
                for name in PROMPT_COLORS_RGB.keys()
            }

            for group in prompt_groups:
                group_prompts = tuple(prompt for _name, prompt in group["prompt_specs"])
                t0 = time.perf_counter()
                trt_inputs = build_static_batch_inputs(
                    processor,
                    images=[image],
                    static_batch=runner.static_batch,
                    text_seq_len=runner.text_seq_len,
                    prompts=group_prompts,
                    include_text=("input_ids" in runner.in_names or "attention_mask" in runner.in_names),
                )
                t1 = time.perf_counter()
                outputs = runner.infer(
                    {k: v for k, v in trt_inputs.items() if k in runner.in_names},
                    profile=args.profile,
                    score_thr=args.score_thr,
                    selective_mask_d2h=not args.no_selective_mask_d2h,
                )
                t2 = time.perf_counter()
                if args.profile:
                    pred_masks_logits, semantic_seg, logit, prof = outputs
                    for key, value in prof.items():
                        if isinstance(value, (int, float)):
                            prof_acc[key] += value
                else:
                    pred_masks_logits, semantic_seg, logit = outputs

                group_results = decode_prompt_group_results(
                    image,
                    pred_masks_logits,
                    logit,
                    group["prompt_specs"],
                    score_thr=args.score_thr,
                    mask_thr=args.mask_thr,
                    process_area_thr=args.area_thr,
                    process_prefer=args.prefer,
                    suppress_conveyor=not args.no_conveyor_suppress,
                    conveyor_dilate=args.conveyor_dilate,
                    mask_nms=not args.no_mask_nms,
                    mask_nms_iou=args.mask_nms_iou,
                    mask_nms_overlap=args.mask_nms_overlap,
                )
                t3 = time.perf_counter()
                pre_ms += (t1 - t0) * 1000.0
                inf_ms += (t2 - t1) * 1000.0
                post_ms += (t3 - t2) * 1000.0

                for category in group["use_categories"]:
                    if category in group_results:
                        append_prompt_result(merged_results[category], group_results[category])

            t4 = time.perf_counter()
            row = finalize_prompt_results_row(
                image,
                image_path,
                merged_results,
                out_dir=args.out_dir,
                category_min_scores=category_min_scores_from_args(args),
                merge_nms_iou=args.merge_nms_iou,
                mask_thr=args.mask_thr,
                top_k=args.top_k,
                suppress_conveyor=not args.no_conveyor_suppress,
                mask_nms=not args.no_mask_nms,
                mask_nms_iou=args.mask_nms_iou,
                mask_nms_overlap=args.mask_nms_overlap,
                save_overlay=not args.no_save_overlays,
                quiet=args.quiet,
            )
            t5 = time.perf_counter()
            post_ms += (t5 - t4) * 1000.0

            timing = {
                "preprocess": pre_ms,
                "inference": inf_ms,
                "postprocess_save": post_ms,
                "total": (t5 - image_t0) * 1000.0,
            }
            timing_rows.append(timing)
            msg = (
                f"[{idx}/{len(image_paths)}] {os.path.basename(image_path)} "
                f"time_ms preprocess={timing['preprocess']:.2f} "
                f"infer={timing['inference']:.2f} "
                f"post={timing['postprocess_save']:.2f} "
                f"total={timing['total']:.2f}"
            )
            if args.profile:
                msg += (
                    f" trt_cuda_total={prof_acc['ms_total_cuda']:.2f} "
                    f"h2d={prof_acc['ms_h2d']:.2f} exec={prof_acc['ms_exec']:.2f} "
                    f"d2h={prof_acc['ms_d2h']:.2f}"
                )
            if not args.quiet:
                print(msg)

            row["timing_ms"] = timing
            if args.profile:
                row["trt_profile_ms"] = {
                    "total_cuda": prof_acc["ms_total_cuda"],
                    "h2d": prof_acc["ms_h2d"],
                    "exec": prof_acc["ms_exec"],
                    "d2h": prof_acc["ms_d2h"],
                    "tail": prof_acc["ms_tail"],
                    "total_cpu": prof_acc["ms_total_cpu"],
                    "copied_masks": prof_acc["copied_masks"],
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_rows_all.append(row)

    print(f"saved predictions: {pred_jsonl_path}")
    if timing_rows:
        avg = {
            key: sum(row[key] for row in timing_rows) / len(timing_rows)
            for key in ("preprocess", "inference", "postprocess_save", "total")
        }
        print(
            "avg_time_ms "
            f"preprocess={avg['preprocess']:.2f} "
            f"infer={avg['inference']:.2f} "
            f"post={avg['postprocess_save']:.2f} "
            f"total={avg['total']:.2f}"
        )

    if not args.no_eval:
        if not os.path.exists(args.annotations):
            print(f"eval skipped: annotation file not found: {args.annotations}")
        else:
            gt_by_file = load_annotations_jsonl(args.annotations)
            eval_y_min = None if args.no_eval_y_filter else args.crop_y_min
            eval_y_max = None if args.no_eval_y_filter else args.crop_y_max
            metrics = evaluate_predictions(
                gt_by_file,
                pred_rows_all,
                iou_thr=args.eval_iou_thr,
                y_min=eval_y_min,
                y_max=eval_y_max,
                y_filter_mode=args.crop_y_mode,
            )
            metrics_path = os.path.join(args.out_dir, "eval_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print_eval_metrics(metrics)
            print(f"saved eval metrics: {metrics_path}")


if __name__ == "__main__":
    main()
