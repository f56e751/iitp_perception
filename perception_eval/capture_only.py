"""Capture color frames from a USB-connected RealSense camera at a chosen
interval and save them as JPGs in <output-dir>/images/.

Pairs with eval_detector.py: this script writes the layout that the
evaluator (and iitp_object_detector.py -i) expects.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output-dir",
        default="tmp_results/perception_eval_260520",
        help="Frames go to <output-dir>/images/.",
    )
    p.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between saved frames (default: 1.0).",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Auto-stop after N seconds (default: run until Ctrl+C).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    images_dir = Path(args.output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    pipeline.start(cfg)

    stop = {"flag": False}

    def _handler(signum, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    duration_msg = (
        f"; auto-stop after {args.duration:.1f}s" if args.duration else "; Ctrl+C to stop"
    )
    print(
        f"capturing to {images_dir} every {args.interval:.2f}s{duration_msg}",
        flush=True,
    )

    saved = 0
    last_write = 0.0
    start_time = time.time()
    try:
        while not stop["flag"]:
            if args.duration is not None and (time.time() - start_time) >= args.duration:
                break
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            now = time.time()
            if now - last_write < args.interval:
                continue
            last_write = now

            img = np.asanyarray(color.get_data())
            path = images_dir / f"{saved:06d}.jpg"
            cv2.imwrite(str(path), img)
            saved += 1
            print(
                f"[{time.strftime('%H:%M:%S')}] saved {path} ({saved} total)",
                flush=True,
            )
    finally:
        pipeline.stop()
        print(f"stopped. {saved} frames in {images_dir}.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
