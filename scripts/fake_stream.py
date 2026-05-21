"""Fake detection producer for testing receivers (no camera / model needed).

Starts the same HTTP server as main.py (streaming.py) and publishes synthetic
detection records on a loop, so you can verify perception_client.py /
recv_detections.py end to end on any machine -- stdlib only, no torch/RealSense.

    python3 scripts/fake_stream.py                 # serve dummy stream on :8080
    # then, on the consumer side:
    python3 scripts/perception_client.py --url http://127.0.0.1:8080/detections/stream
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streaming

CLASSES = ["transparent", "metal", "cardboard"]


def random_detections(max_objects, rng=random):
    """Return (positions, class_names, confidences) parallel arrays of equal length."""
    n = rng.randint(0, max_objects)
    positions = [
        (
            round(rng.uniform(-0.5, 0.5), 3),
            round(rng.uniform(-0.4, 0.4), 3),
            round(rng.uniform(0.3, 2.0), 3),
        )
        for _ in range(n)
    ]
    class_names = [rng.choice(CLASSES) for _ in range(n)]
    confidences = [round(rng.uniform(0.2, 0.95), 3) for _ in range(n)]
    return positions, class_names, confidences


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--interval", type=float, default=0.2, help="Seconds between frames.")
    p.add_argument("--max-objects", type=int, default=3, help="Max objects per frame.")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducible output.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    streaming.start_server(args.port, fps=30)
    print(
        f"fake detection stream on :{args.port}  "
        f"(/detections latest, /detections/stream live) -- Ctrl+C to stop",
        flush=True,
    )
    try:
        while True:
            positions, class_names, confidences = random_detections(args.max_objects, rng)
            record = streaming.build_record(
                time.time(), args.interval, positions, class_names, confidences
            )
            streaming.publish_detections(record)
            print(f"published {len(class_names)} objs: {class_names}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("stopped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
