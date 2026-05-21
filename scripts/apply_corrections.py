"""Apply manual corrections to tracking output.

Reads <input-dir>/objects/corrections.json (or a path passed via --corrections),
rebuilds the per-track CSVs, summary.csv, and annotated_tracks/ frames from
scores.csv plus the corrected (frame, box_id) -> track_id mapping.

corrections.json schema:
{
  "delete_tracks": [16, 18],
  "delete_frames": ["000040.jpg"],
  "reassignments": [
    {"frame": "000039.jpg", "box_id": 0, "to_track": 19, "reason": "..."}
  ],
  "merge_tracks": [
    {"target": 15, "sources": [20],
     "detection_failures": ["000034.jpg"], "reason": "..."}
  ]
}

Idempotent: re-running produces the same output, so you can iterate on
the corrections file freely.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import cv2

PALETTE = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0),
    (255, 0, 255), (0, 128, 255), (128, 0, 255), (128, 255, 0), (255, 128, 128),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i", "--input-dir",
        default="tmp_results/perception_eval_260520",
    )
    p.add_argument(
        "--corrections",
        default=None,
        help="Path to corrections.json. Defaults to <input-dir>/objects/corrections.json.",
    )
    p.add_argument(
        "--no-annotate", action="store_true",
        help="Skip re-rendering annotated_tracks/.",
    )
    return p.parse_args()


def load_original_assignments(objects_dir: Path):
    """Read existing track_*.csv files into {(image, box_id): track_id}."""
    assignments = {}
    pattern = re.compile(r"track_(\d+)\.csv$")
    for path in objects_dir.glob("track_*.csv"):
        m = pattern.search(path.name)
        if not m:
            continue
        tid = int(m.group(1))
        with path.open() as f:
            for row in csv.DictReader(f):
                assignments[(row["image"], row["box_id"])] = tid
    return assignments


def centroid(row):
    x1, y1 = float(row["x1"]), float(row["y1"])
    x2, y2 = float(row["x2"]), float(row["y2"])
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def render(input_dir: Path, by_frame_with_tid: dict, frame_names: list):
    images_dir = input_dir / "images"
    out_dir = input_dir / "annotated_tracks"
    out_dir.mkdir(exist_ok=True)
    for p in out_dir.glob("*.jpg"):
        p.unlink()
    n = 0
    for fname in frame_names:
        img_path = images_dir / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for row, tid in by_frame_with_tid.get(fname, []):
            if tid is None:
                continue
            x1 = int(float(row["x1"])); y1 = int(float(row["y1"]))
            x2 = int(float(row["x2"])); y2 = int(float(row["y2"]))
            color = PALETTE[tid % len(PALETTE)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"T{tid} {row['predicted_class']} {float(row['top_score']):.2f}"
            text_y = y1 - 6 if y1 > 18 else y1 + 14
            cv2.putText(img, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / fname), img)
        n += 1
    print(f"wrote {n} corrected annotated frames to {out_dir}", flush=True)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    objects_dir = input_dir / "objects"
    scores_path = input_dir / "scores.csv"
    corrections_path = (
        Path(args.corrections) if args.corrections else objects_dir / "corrections.json"
    )

    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found.", file=sys.stderr)
        return 1
    if not corrections_path.exists():
        print(f"ERROR: {corrections_path} not found.", file=sys.stderr)
        return 1

    corrections = json.loads(corrections_path.read_text())
    assignments = load_original_assignments(objects_dir)

    # Apply deletions (whole tracks).
    deleted = set(corrections.get("delete_tracks", []))
    for key in list(assignments.keys()):
        if assignments[key] in deleted:
            del assignments[key]

    # Apply deletions (entire frames - e.g., user held the object in their hand).
    deleted_frames = set(corrections.get("delete_frames", []))
    for key in list(assignments.keys()):
        image, _ = key
        if image in deleted_frames:
            del assignments[key]

    # Apply per-detection reassignments.
    for r in corrections.get("reassignments", []):
        key = (r["frame"], str(r["box_id"]))
        if key not in assignments:
            print(f"WARN: reassignment for {key} ignored (not in any track).", file=sys.stderr)
            continue
        assignments[key] = r["to_track"]

    # Apply merges (sources -> target). Collect missed_frames per track.
    notes: dict[int, dict] = {}
    for m in corrections.get("merge_tracks", []):
        target = m["target"]
        for src in m.get("sources", []):
            for key, val in list(assignments.items()):
                if val == src:
                    assignments[key] = target
        misses = m.get("detection_failures", [])
        if misses:
            notes.setdefault(target, {}).setdefault("missed_frames", []).extend(misses)

    # Rebuild tracks from scores.csv via the corrected mapping.
    with scores_path.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        score_rows = list(reader)

    tracks: dict[int, list[dict]] = {}
    by_frame_with_tid: dict[str, list[tuple]] = {}
    for row in score_rows:
        key = (row["image"], row["box_id"])
        tid = assignments.get(key)
        by_frame_with_tid.setdefault(row["image"], []).append((row, tid))
        if tid is None:
            continue
        tracks.setdefault(tid, []).append(row)

    # Wipe old per-track CSVs, write new ones + new summary.
    for p in objects_dir.glob("track_*.csv"):
        p.unlink()
    summary_path = objects_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    track_header = ["track_id"] + list(header)
    with summary_path.open("w", newline="") as fsum:
        sw = csv.writer(fsum)
        sw.writerow([
            "track_id", "kind", "n_frames", "first_frame", "last_frame",
            "mean_x_center", "mean_y_center", "max_motion_px",
            "dominant_class", "class_counts", "missed_frames",
        ])

        for tid in sorted(tracks):
            tr_rows = tracks[tid]
            # Order rows by image filename (frames are zero-padded so lexical sort = temporal).
            tr_rows.sort(key=lambda r: r["image"])
            centroids = [centroid(r) for r in tr_rows]
            n = len(tr_rows)
            max_motion = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(centroids[i], centroids[j])
                    if d > max_motion:
                        max_motion = d
            mean_x = sum(c[0] for c in centroids) / n
            mean_y = sum(c[1] for c in centroids) / n
            counts = Counter(r["predicted_class"] for r in tr_rows)
            dominant = counts.most_common(1)[0][0]
            kind = "moving" if max_motion >= 30 else "static"
            missed = notes.get(tid, {}).get("missed_frames", [])

            with (objects_dir / f"track_{tid}.csv").open("w", newline="") as ftr:
                tw = csv.DictWriter(ftr, fieldnames=track_header)
                tw.writeheader()
                for r in tr_rows:
                    out = {"track_id": tid}
                    out.update(r)
                    tw.writerow(out)

            sw.writerow([
                tid, kind, n, tr_rows[0]["image"], tr_rows[-1]["image"],
                f"{mean_x:.1f}", f"{mean_y:.1f}", f"{max_motion:.1f}",
                dominant, dict(counts), ";".join(missed),
            ])
            print(
                f"track_{tid}: {kind}, {n} frames, "
                f"{tr_rows[0]['image']}..{tr_rows[-1]['image']}, "
                f"counts={dict(counts)}"
                + (f", missed={missed}" if missed else ""),
                flush=True,
            )

    if not args.no_annotate:
        frame_names = sorted(by_frame_with_tid.keys())
        render(input_dir, by_frame_with_tid, frame_names)

    return 0


if __name__ == "__main__":
    sys.exit(main())
