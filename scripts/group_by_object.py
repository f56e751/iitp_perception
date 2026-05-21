"""Split scores.csv into per-object CSVs by grouping detections with high
IoU across frames.

Assumes a roughly static scene (objects don't move much between frames) —
each cluster's representative box stays the one we first saw it as.

Output layout (under --input-dir):
  objects/
    object_0.csv  # leftmost cluster (by mean x-center)
    object_1.csv
    ...
"""

import argparse
import csv
import sys
from pathlib import Path


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input-dir",
        default="tmp_results/perception_eval_260520",
        help="Folder containing scores.csv.",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Min IoU to consider two boxes the same object (default: 0.5).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    csv_path = input_dir / "scores.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist.", file=sys.stderr)
        return 1

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    # Greedy IoU clustering. Each cluster: representative box + accumulated rows.
    clusters = []  # list[dict(box=tuple, rows=list[dict])]
    for row in rows:
        box = (
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        )
        best_idx, best_iou = -1, 0.0
        for i, c in enumerate(clusters):
            score = iou(box, c["box"])
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_iou >= args.iou_threshold:
            clusters[best_idx]["rows"].append(row)
        else:
            clusters.append({"box": box, "rows": [row]})

    # Sort clusters left-to-right by mean x-center for stable IDs.
    def center_x(c):
        xs = [(float(r["x1"]) + float(r["x2"])) / 2 for r in c["rows"]]
        return sum(xs) / len(xs)

    clusters.sort(key=center_x)

    objects_dir = input_dir / "objects"
    objects_dir.mkdir(exist_ok=True)
    # Wipe any stale per-object CSVs from a previous run.
    for p in objects_dir.glob("object_*.csv"):
        p.unlink()

    for i, c in enumerate(clusters):
        out_path = objects_dir / f"object_{i}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(c["rows"])
        cx = center_x(c)
        cy = sum(
            (float(r["y1"]) + float(r["y2"])) / 2 for r in c["rows"]
        ) / len(c["rows"])
        classes = [r["predicted_class"] for r in c["rows"]]
        counts = {cls: classes.count(cls) for cls in set(classes)}
        print(
            f"object_{i}: {len(c['rows'])} frames, "
            f"center≈({cx:.0f},{cy:.0f}), class counts={counts} -> {out_path}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
