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


def _publish_frame(annotated_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(
        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
    )
    if ok:
        streaming.publish_frame(bytes(buf))


# Annotators for the live stream. Drawing happens on the FULL frame (after the
# sides are blurred) so a box label near the crop edge can spill over the
# blurred region instead of being clipped. text_scale mirrors the engine.
_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
_LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)


def _compose_stream_frame(
    full_bgr: np.ndarray, center_bgr: np.ndarray, x0: int, x1: int
) -> np.ndarray:
    """Full-size frame: trimmed side quarters blurred, raw center dropped in,
    then detection boxes/labels drawn on top in full-frame coordinates.

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
        out = _LABEL_ANNOTATOR.annotate(
            scene=out, detections=shifted, labels=labels
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
    # Cropping shifts the principal point left by crop_x0 (fx/fy/cy unchanged),
    # so 3D positions stay in true camera coordinates.
    intr_crop = (fx, fy, cx - crop_x0, cy)
    print(
        f"inference crop: cols [{crop_x0}:{crop_x1}] of {COLOR_W} "
        f"(cx {cx:.1f} -> {cx - crop_x0:.1f})",
        flush=True,
    )

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
                model, color_crop, depth_crop, intr_crop
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
