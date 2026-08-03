"""Production entry point: RealSense capture -> detection -> stream out.

The camera is plugged directly into this machine and frames are grabbed via
pyrealsense2 (no network round-trip from a remote robot PC).

Detections are appended one JSON record per frame to results_local/detections.jsonl
and pushed live to other computers over HTTP (see streaming.py):
    http://<host>:8080/stream             annotated MJPEG video
    http://<host>:8080/detections         latest detection record (JSON)
    http://<host>:8080/detections/stream  live NDJSON detection stream
"""

import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
import torch

import iitp_object_detector
import streaming
from capture_timing import capture_age_seconds
from grounding_dino.groundingdino.util.inference import load_model
from iitp_object_detector import object_detector


DEVICE = torch.device("cuda:0")
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "checkpoint_best.pth"

OUTPUT_DIR = Path("results_local")
OUTPUT_DIR.mkdir(exist_ok=True)
JSONL_PATH = OUTPUT_DIR / "detections.jsonl"

COLOR_W, COLOR_H, FPS = 640, 480, 30
STREAM_PORT = 8080
STREAM_JPEG_QUALITY = 80

# Trim this fraction of the width off EACH side before inference, keeping only
# the center column (default 1/4 + 1/4 trimmed -> center half fed to the model).
# The trimmed sides are not discarded from the stream: they are shown blurred.
SIDE_CROP_FRAC = 0.25
BLUR_SIGMA = 15  # Gaussian sigma for the blurred side quarters in the stream

# Calibration for the depth-free pixel-ratio projection: the real-world length
# (m) that the FULL image height (COLOR_H pixels) spans on the working plane.
# X and Y are then computed as pixel offsets from the image center scaled by
# the same metres-per-pixel (square-pixel assumption: fx ~= fy on D455).
# Origin = image center; +X right, +Y down (camera frame). Z is not measured.
# Calibrated from a 50 cm-wide box spanning 297 px -> 0.16835 cm/px
# (0.16835 cm/px * 480 px = 80.8 cm full-height span).
VISIBLE_Y_LENGTH_M = 0.808

# Real-world coordinate grid drawn (thin) on the stream, in cm.
GRID_STEP_CM = 5          # spacing between grid lines
GRID_LABEL_EVERY_CM = 10  # label only every Nth line to avoid clutter


def _publish_frame(annotated_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(
        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
    )
    if ok:
        streaming.publish_frame(bytes(buf))


# Annotators for the live stream. Two supervision LabelAnnotators give the
# class line and the coordinate line the same filled-background "CSS" style;
# the coord line just uses a smaller text scale.
_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
_CLASS_PADDING = 10
_COORD_PADDING = 5
_LABEL_ANNOTATOR_CLASS = sv.LabelAnnotator(
    text_thickness=1, text_scale=0.5, text_padding=_CLASS_PADDING
)
_LABEL_ANNOTATOR_COORD = sv.LabelAnnotator(
    text_thickness=1, text_scale=0.3, text_padding=_COORD_PADDING
)


def _label_box_height(text_scale: float, text_padding: int) -> int:
    """Match supervision's label-box height = text_height + 2*text_padding."""
    (_, text_h), _ = cv2.getTextSize(
        "Ag", cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1
    )
    return text_h + 2 * text_padding


_COORD_LABEL_H = _label_box_height(0.3, _COORD_PADDING)


def _build_grid(px_to_cm, cm_to_px, x0, x1, height, step_cm):
    """Precompute pixel line segments for a real-world `step_cm` grid covering
    the cm extent visible in the inference column [x0, x1).

    Returns (segments, clip_rect); each segment is (value_cm, axis, p0, p1)
    with axis "x" (constant X, vertical-ish) or "y" (constant Y).
    """
    import math
    corners = [(x0, 0), (x1, 0), (x1, height), (x0, height)]
    xs, ys = zip(*(px_to_cm(u, v) for u, v in corners))
    x_lo = math.floor(min(xs) / step_cm) * step_cm
    x_hi = math.ceil(max(xs) / step_cm) * step_cm
    y_lo = math.floor(min(ys) / step_cm) * step_cm
    y_hi = math.ceil(max(ys) / step_cm) * step_cm

    segs = []
    val = x_lo
    while val <= x_hi + 1e-6:
        segs.append((val, "x", cm_to_px(val, y_lo), cm_to_px(val, y_hi)))
        val += step_cm
    val = y_lo
    while val <= y_hi + 1e-6:
        segs.append((val, "y", cm_to_px(x_lo, val), cm_to_px(x_hi, val)))
        val += step_cm
    return segs, (int(x0), 0, int(x1 - x0), int(height))


def _draw_grid(out: np.ndarray, grid) -> None:
    """Draw the precomputed cm grid, clipped to the inference column."""
    segs, rect = grid
    for value, axis, p0, p1 in segs:
        q0 = (int(round(p0[0])), int(round(p0[1])))
        q1 = (int(round(p1[0])), int(round(p1[1])))
        ok, a, b = cv2.clipLine(rect, q0, q1)
        if not ok:
            continue
        is_axis = abs(value) < 1e-6
        color = (0, 215, 255) if is_axis else (110, 110, 110)
        cv2.line(out, a, b, color, 1, cv2.LINE_AA)
        if round(value) % GRID_LABEL_EVERY_CM == 0:
            if axis == "x":
                lx, ly = a if a[1] < b[1] else b
                cv2.putText(out, f"{value:+.0f}", (lx + 2, ly + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
            else:
                lx, ly = a if a[0] < b[0] else b
                cv2.putText(out, f"{value:+.0f}", (lx + 2, ly - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)


def _compose_stream_frame(
    full_bgr: np.ndarray, center_bgr: np.ndarray, x0: int, x1: int, grid=None
) -> np.ndarray:
    """Full-size frame: trimmed side quarters blurred, raw center dropped in,
    then the cm grid and detection boxes/labels drawn on top in full-frame coords.

    `center_bgr` is the raw center crop spanning columns [x0:x1) and the full
    height. Boxes come from the detector in crop coords and are shifted by x0,
    so labels render on top of everything — including the blurred sides.
    """
    out = full_bgr.copy()
    out[:, :x0] = cv2.GaussianBlur(out[:, :x0], (0, 0), BLUR_SIGMA)
    out[:, x1:] = cv2.GaussianBlur(out[:, x1:], (0, 0), BLUR_SIGMA)
    out[:, x0:x1] = center_bgr

    if grid is not None:
        _draw_grid(out, grid)

    det = iitp_object_detector.LAST_DETECTIONS
    labels = iitp_object_detector.LAST_LABELS
    if det is not None and len(det) > 0:
        xyxy = det.xyxy.copy()
        xyxy[:, [0, 2]] += x0  # crop pixel coords -> full-frame coords
        shifted = sv.Detections(xyxy=xyxy, class_id=det.class_id)
        out = _BOX_ANNOTATOR.annotate(scene=out, detections=shifted)

        # Engine builds "<class> <score> (<X>,<Y>,<Z>)m"; split on first " (".
        heads, coords = [], []
        for lbl in labels:
            if " (" in lbl:
                h, t = lbl.split(" (", 1)
                heads.append(h)
                coords.append("(" + t)
            else:
                heads.append(lbl)
                coords.append("")

        # Stack two supervision labels above the box: class on top, coord just
        # above the box. Coord uses the real box (drawn first, sits at box top).
        # Class uses a virtual box shifted up by coord-label-height so it lands
        # right above the coord label.
        out = _LABEL_ANNOTATOR_COORD.annotate(
            scene=out, detections=shifted, labels=coords
        )
        virt_xyxy = xyxy.copy()
        virt_xyxy[:, 1] -= _COORD_LABEL_H  # raise the top so class sits higher
        virt = sv.Detections(xyxy=virt_xyxy, class_id=det.class_id)
        out = _LABEL_ANNOTATOR_CLASS.annotate(
            scene=out, detections=virt, labels=heads
        )
    return out


def main() -> None:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16, FPS)
    profile = pipeline.start(cfg)

    global_time_sensors = 0
    for sensor in profile.get_device().query_sensors():
        try:
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)
                global_time_sensors += 1
        except RuntimeError as exc:
            print(f"warning: could not enable RealSense global time: {exc}", flush=True)
    print(
        f"RealSense global time enabled on {global_time_sensors} sensor(s)",
        flush=True,
    )

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print(
        f"camera intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}",
        flush=True,
    )

    # Center crop fed to the model: trim SIDE_CROP_FRAC off each side.
    crop_x0 = int(round(COLOR_W * SIDE_CROP_FRAC))
    crop_x1 = COLOR_W - crop_x0
    print(
        f"inference crop: cols [{crop_x0}:{crop_x1}] of {COLOR_W}", flush=True
    )

    # Depth-free projection. Prefer a 4-point homography (handles camera tilt)
    # if calibration/homography.json exists; otherwise fall back to the plain
    # pixel-ratio scaled to VISIBLE_Y_LENGTH_M. Z is always 0.0 (not measured).
    # Flat 2D-plane assumption: no camera-tilt / homography correction. Pure
    # linear pixel-ratio mapping (image center origin, +X right, +Y down),
    # scaled so the full image height spans VISIBLE_Y_LENGTH_M. px_to_cm /
    # cm_to_px map FULL-frame pixels <-> real cm.
    cm_per_pixel = (VISIBLE_Y_LENGTH_M / COLOR_H) * 100.0
    img_cx = COLOR_W / 2.0
    img_cy = COLOR_H / 2.0
    print(
        f"projection: flat 2D pixel-ratio, {cm_per_pixel * 10:.3f} mm/px "
        f"(no tilt correction)",
        flush=True,
    )

    def px_to_cm(u_full, v):
        return (u_full - img_cx) * cm_per_pixel, (v - img_cy) * cm_per_pixel

    def cm_to_px(x_cm, y_cm):
        return x_cm / cm_per_pixel + img_cx, y_cm / cm_per_pixel + img_cy

    def project(u_crop, v_crop, _depth_np):
        x_cm, y_cm = px_to_cm(u_crop + crop_x0, v_crop)
        return (x_cm / 100.0, y_cm / 100.0, 0.0)

    # Precompute the cm grid once (drawn on the stream each frame).
    grid = _build_grid(px_to_cm, cm_to_px, crop_x0, crop_x1, COLOR_H, GRID_STEP_CM)

    align = rs.align(rs.stream.color)

    print("loading grounding model...", flush=True)
    model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )
    print(f"writing results to {JSONL_PATH}", flush=True)

    streaming.start_server(STREAM_PORT, FPS)
    print(
        f"streams ready on :{STREAM_PORT}  "
        f"(/stream video, /detections latest, /detections/stream live)",
        flush=True,
    )

    stop = {"flag": False}

    def _handler(signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    f = JSONL_PATH.open("a", buffering=1)
    capture_timestamp_warned = False
    try:
        while not stop["flag"]:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue
            color_np = np.asanyarray(color.get_data())
            depth_np = np.asanyarray(depth.get_data())

            # Feed only the center crop to the model (sides are out of interest).
            color_crop = np.ascontiguousarray(color_np[:, crop_x0:crop_x1])
            depth_crop = np.ascontiguousarray(depth_np[:, crop_x0:crop_x1])

            ts = time.time()
            timestamp_domain = color.get_frame_timestamp_domain()
            frame_timestamp_ms = color.get_timestamp()
            capture_age = capture_age_seconds(
                ts,
                frame_timestamp_ms,
                is_global_time=(timestamp_domain == rs.timestamp_domain.global_time),
            )
            capture_timestamp = (
                frame_timestamp_ms / 1000.0 if capture_age is not None else None
            )
            if capture_age is None and not capture_timestamp_warned:
                print(
                    "warning: unusable RealSense global timestamp; "
                    "consumers will fall back to estimated frame age "
                    f"(domain={timestamp_domain}, timestamp_ms={frame_timestamp_ms:.3f})",
                    flush=True,
                )
                capture_timestamp_warned = True
            bounding_boxes, class_names, confidences = object_detector(
                model, color_crop, depth_crop, project
            )
            elapsed = time.time() - ts

            record = streaming.build_record(
                ts,
                elapsed,
                bounding_boxes,
                class_names,
                confidences,
                capture_timestamp=capture_timestamp,
                capture_age_s=capture_age,
                capture_timestamp_domain=str(timestamp_domain),
            )
            f.write(json.dumps(record) + "\n")
            streaming.publish_detections(record)

            # Stream: full frame with the trimmed side quarters blurred, the raw
            # center dropped in, and detection boxes/labels drawn on top (labels
            # may overlap the blurred sides).
            frame_to_publish = _compose_stream_frame(
                color_np, color_crop, crop_x0, crop_x1, grid
            )
            _publish_frame(frame_to_publish)

            print(
                f"[{time.strftime('%H:%M:%S')}] {len(bounding_boxes)} objs in {elapsed:.2f}s",
                flush=True,
            )
    finally:
        f.close()
        pipeline.stop()
        print("stopped.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
