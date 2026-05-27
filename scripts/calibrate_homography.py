"""Compute a pixel->plane homography for depth-free projection.

Use this when the camera is not perpendicular to the working surface, so the
plain pixel-ratio projection in main.py over/underestimates positions toward
the image edges. A 4-point homography absorbs any camera tilt as long as the
target objects lie on a single plane.

Workflow:
  1. Start the live stream (run main.py on the camera PC).
  2. Place 4 reference points on the working plane whose REAL positions you
     know (e.g., the corners of a rectangle of known size, centered on the
     image centre). The points must be inside the unblurred centre region.
  3. Run this script (any machine that can reach :8080):
         python3 scripts/calibrate_homography.py
     It fetches one snapshot to calibration/snapshot.jpg.
  4. Open the snapshot in any image viewer, read each corner's pixel (u, v).
  5. Type each pixel + the corresponding real-world position in cm
     (image-centre origin: +X right, +Y down) at the prompts.
  6. Tool prints residuals and writes calibration/homography.json. Restart
     main.py to apply -- it auto-detects the file and switches projection.

If the file is removed, main.py falls back to the plain pixel-ratio mode.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OUT = Path("calibration/homography.json")
DEFAULT_SNAPSHOT = Path("calibration/snapshot.jpg")
NUM_POINTS = 4


def fetch_snapshot(url: str, out_path: Path, timeout: float = 8.0) -> None:
    """Pull one JPEG frame out of the MJPEG /stream and save it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = b""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        while len(data) < 600_000:
            chunk = r.read(4096)
            if not chunk:
                break
            data += chunk
            start = data.find(b"\xff\xd8")
            end = data.find(b"\xff\xd9", start + 2)
            if start != -1 and end != -1:
                out_path.write_bytes(data[start:end + 2])
                return
    raise RuntimeError("could not extract a JPEG frame from the stream")


def prompt_points(n: int):
    """Collect n (pixel, real_cm) calibration pairs from stdin."""
    print(
        f"\nEnter {n} calibration points. For each:\n"
        f"  - pixel u v (read from the snapshot in your image viewer)\n"
        f"  - real X Y in cm relative to the IMAGE CENTRE (+X right, +Y down)\n"
        f"Type two numbers separated by a space at each prompt.\n"
    )
    points = []
    for i in range(1, n + 1):
        while True:
            try:
                u, v = (float(x) for x in input(f"Point {i} pixel u v: ").split())
                X, Y = (float(x) for x in input(f"Point {i} real cm X Y: ").split())
                points.append({"pixel": [u, v], "real_cm": [X, Y]})
                break
            except (ValueError, EOFError):
                print("  (need two numeric values separated by a space; try again)")
    return points


def compute_homography(points):
    src = np.array([p["pixel"] for p in points], dtype=np.float32)
    dst = np.array([p["real_cm"] for p in points], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def report_residuals(points, H: np.ndarray) -> float:
    """Print per-point residual; return the max |error| in cm."""
    print("\nresiduals (pixel -> projected vs measured cm):")
    max_err = 0.0
    for p in points:
        u, v = p["pixel"]
        h = H @ np.array([u, v, 1.0])
        Xp, Yp = h[0] / h[2], h[1] / h[2]
        ex = Xp - p["real_cm"][0]
        ey = Yp - p["real_cm"][1]
        err = (ex * ex + ey * ey) ** 0.5
        max_err = max(max_err, err)
        print(
            f"  ({u:6.1f},{v:6.1f}) -> ({Xp:+7.2f},{Yp:+7.2f})  "
            f"vs ({p['real_cm'][0]:+7.2f},{p['real_cm'][1]:+7.2f})  "
            f"err ({ex:+.3f},{ey:+.3f})  |{err:.3f}| cm"
        )
    return max_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream-url", default="http://127.0.0.1:8080/stream",
                    help="MJPEG endpoint to grab the snapshot from.")
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-snapshot", action="store_true",
                    help="Reuse the existing snapshot at --snapshot.")
    args = ap.parse_args()

    if not args.skip_snapshot:
        print(f"fetching snapshot from {args.stream_url} ...")
        fetch_snapshot(args.stream_url, args.snapshot)
        print(f"saved snapshot to {args.snapshot}")
        print("open it in your image viewer and note pixel (u, v) for each known point.")

    points = prompt_points(NUM_POINTS)
    H = compute_homography(points)
    max_err = report_residuals(points, H)
    if max_err > 0.01:
        print(
            f"\nWARNING: 4-point homography should fit with ~0 residual; max err "
            f"is {max_err:.3f} cm -- recheck your inputs (typos in pixel or cm?)."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "homography_v1",
        "image_size_pixels": [640, 480],
        "unit": "cm",
        "origin": "image_centre",
        "calibration_points": points,
        "homography": H.tolist(),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    print("restart main.py to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
