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
VISIBLE_Y_LENGTH_M = 0.78


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


def _compose_stream_frame(
    full_bgr: np.ndarray, center_bgr: np.ndarray, x0: int, x1: int
) -> np.ndarray:
    """Full-size frame: trimmed side quarters blurred, raw center dropped in,
    then detection boxes and stacked labels drawn on top in full-frame coords.

    `center_bgr` is the raw center crop spanning columns [x0:x1) and the full
    height. Boxes come from the detector in crop coords and are shifted by x0,
    so labels render on top of everything — including the blurred sides.
    """
    out = full_bgr.copy()
    out[:, :x0] = cv2.GaussianBlur(out[:, :x0], (0, 0), BLUR_SIGMA)
    out[:, x1:] = cv2.GaussianBlur(out[:, x1:], (0, 0), BLUR_SIGMA)
    out[:, x0:x1] = center_bgr

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
    homography_path = Path("calibration/homography.json")
    if homography_path.exists():
        cal = json.loads(homography_path.read_text())
        H_cm = np.array(cal["homography"], dtype=np.float64)
        print(
            f"projection: homography from {homography_path} "
            f"(calibrated on {len(cal.get('calibration_points', []))} points)",
            flush=True,
        )

        def project(u_crop, v_crop, _depth_np):
            u_full = u_crop + crop_x0
            h = H_cm @ np.array([u_full, v_crop, 1.0])
            X_cm, Y_cm = h[0] / h[2], h[1] / h[2]
            return (X_cm / 100.0, Y_cm / 100.0, 0.0)
    else:
        m_per_pixel = VISIBLE_Y_LENGTH_M / COLOR_H
        img_cx = COLOR_W / 2.0
        img_cy = COLOR_H / 2.0
        print(
            f"projection: pixel-ratio, {m_per_pixel * 1000:.3f} mm/px "
            f"(image span {COLOR_W * m_per_pixel:.3f} x {COLOR_H * m_per_pixel:.3f} m) "
            f"-- run scripts/calibrate_homography.py to tilt-correct",
            flush=True,
        )

        def project(u_crop, v_crop, _depth_np):
            u_full = u_crop + crop_x0
            X = (u_full - img_cx) * m_per_pixel
            Y = (v_crop - img_cy) * m_per_pixel
            return (X, Y, 0.0)

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
            positions, class_names, confidences = object_detector(
                model, color_crop, depth_crop, project
            )
            elapsed = time.time() - ts

            record = streaming.build_record(
                ts, elapsed, positions, class_names, confidences
            )
            f.write(json.dumps(record) + "\n")
            streaming.publish_detections(record)

            # Stream: full frame with the trimmed side quarters blurred, the raw
            # center dropped in, and detection boxes/labels drawn on top (labels
            # may overlap the blurred sides).
            frame_to_publish = _compose_stream_frame(
                color_np, color_crop, crop_x0, crop_x1
            )
            _publish_frame(frame_to_publish)

            print(
                f"[{time.strftime('%H:%M:%S')}] {len(positions)} objs in {elapsed:.2f}s",
                flush=True,
            )
    finally:
        f.close()
        pipeline.stop()
        print("stopped.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
