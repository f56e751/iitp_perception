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


def _publish_frame(annotated_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(
        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
    )
    if ok:
        streaming.publish_frame(bytes(buf))


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

            ts = time.time()
            positions, class_names, confidences = object_detector(
                model, color_np, depth_np, (fx, fy, cx, cy)
            )
            elapsed = time.time() - ts

            record = streaming.build_record(
                ts, elapsed, positions, class_names, confidences
            )
            f.write(json.dumps(record) + "\n")
            streaming.publish_detections(record)

            # Publish latest frame to MJPEG stream. Fall back to the raw
            # color frame when no detections (annotator was not invoked).
            frame_to_publish = iitp_object_detector.LAST_ANNOTATED
            if frame_to_publish is None:
                frame_to_publish = color_np
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
