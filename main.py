"""Production entry point: RealSense capture -> detection -> MJPEG stream.

The camera is plugged directly into this machine and frames are grabbed via
pyrealsense2 (no network round-trip from a remote robot PC).

Detections are appended one JSON record per frame to results_local/detections.jsonl.
A background thread serves the latest annotated frame at
    http://<host>:8080/stream
as MJPEG (multipart/x-mixed-replace) — open in any browser.
"""

import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

import iitp_object_detector
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

_latest = {"jpeg": None}
_latest_lock = threading.Lock()


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_a, **_kw):
        pass

    def do_GET(self):
        if self.path in ("/", "/stream"):
            self._serve_stream()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        try:
            while True:
                with _latest_lock:
                    data = _latest["jpeg"]
                if data is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(data)}\r\n\r\n".encode()
                )
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _start_stream_server() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", STREAM_PORT), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"MJPEG stream ready: http://<this-host>:{STREAM_PORT}/stream",
        flush=True,
    )


def _publish_frame(annotated_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(
        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
    )
    if not ok:
        return
    data = bytes(buf)
    with _latest_lock:
        _latest["jpeg"] = data


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

    _start_stream_server()

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
            positions, class_names = object_detector(
                model, color_np, depth_np, (fx, fy, cx, cy)
            )
            elapsed = time.time() - ts

            record = {
                "timestamp": ts,
                "elapsed_s": elapsed,
                "positions": [[float(v) for v in p] for p in positions],
                "class_names": list(class_names),
            }
            f.write(json.dumps(record) + "\n")

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
