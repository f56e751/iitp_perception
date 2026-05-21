"""Receive the live detection stream on another computer (stdlib only).

Connects to main.py's NDJSON endpoint and prints one detection record per
line as it arrives. No torch / camera / extra packages needed -- run this on
the robot PC (or any machine on the LAN) as a starting point for consuming
the perception output.

    python3 recv_detections.py --url http://<server-ip>:8080/detections/stream
"""
import argparse
import json
import sys
import urllib.request


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default="http://147.46.175.15:8080/detections/stream",
        help="NDJSON detection stream URL exposed by main.py.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"connecting to {args.url} ...", flush=True)
    with urllib.request.urlopen(args.url) as resp:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Replace this with your robot-side handling (e.g. publish poses).
            n = len(rec.get("positions", []))
            print(
                f"[{rec.get('timestamp', 0):.3f}] {n} objs "
                f"classes={rec.get('class_names')} "
                f"conf={[round(c, 2) for c in rec.get('confidences', [])]}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
