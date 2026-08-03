"""Build a Grounding-DINO finetune dataset from curated perception runs.

Reads curated runs under `<batch>/run_*/` (each containing `images/`,
`scores.csv`, `objects/summary.csv`, `objects/track_*.csv`,
`objects/true_labels.json`, and `objects/corrections.json`), and emits
an ODVG-detection JSONL dataset suitable for finetuning Grounding-DINO.

Excludes any frame with a known data-quality issue: detector misses,
deleted erroneous detections, per-detection deletes, or whole-frame
deletes.

Output structure:

    <out>/
      train/
        images/000000.jpg ...
        annotations.jsonl
      val/
        images/000000.jpg ...
        annotations.jsonl
      categories.json
      README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


CATEGORIES = [
    {"id": 0, "name": "transparent"},
    {"id": 1, "name": "metal"},
    {"id": 2, "name": "cardboard"},
]
CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}

DEFAULT_BATCH = Path(
    "perception_tests/results/perception_eval_260524"
)
DEFAULT_TRAIN = [f"run_{i}" for i in range(1, 18)]
DEFAULT_VAL = [f"run_{i}" for i in range(101, 106)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", required=True,
                   help="Dataset root directory (will be created).")
    p.add_argument("--batch", default=str(DEFAULT_BATCH),
                   help="Source batch directory containing run_*/ folders.")
    p.add_argument("--train", nargs="*", default=DEFAULT_TRAIN,
                   help="Run names for the train split.")
    p.add_argument("--val", nargs="*", default=DEFAULT_VAL,
                   help="Run names for the val split.")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink images instead of copying.")
    return p.parse_args()


def excluded_frames(run_dir: Path) -> set[str]:
    """Frames excluded from the dataset for this run."""
    excl: set[str] = set()
    summary = run_dir / "objects" / "summary.csv"
    if summary.exists():
        with summary.open() as f:
            for row in csv.DictReader(f):
                missed = row.get("missed_frames", "") or ""
                for frame in missed.split(";"):
                    if frame:
                        excl.add(frame)
    corrections = run_dir / "objects" / "corrections.json"
    if corrections.exists():
        d = json.loads(corrections.read_text())
        notes = d.get("_delete_track_notes", {})
        for v in notes.values():
            if isinstance(v, dict) and "frame" in v:
                excl.add(v["frame"])
        for x in d.get("delete_detections", []):
            excl.add(x["frame"])
        for frame in d.get("delete_frames", []):
            excl.add(frame)
    return excl


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(float(x1), float(w)))
    x2 = max(0.0, min(float(x2), float(w)))
    y1 = max(0.0, min(float(y1), float(h)))
    y2 = max(0.0, min(float(y2), float(h)))
    return x1, y1, x2, y2


def collect_run_instances(run_dir: Path):
    """Returns {frame_name: [(bbox, category), ...]} for all labeled
    detections in `run_dir`, excluding bad-quality frames."""
    labels_path = run_dir / "objects" / "true_labels.json"
    if not labels_path.exists():
        print(f"WARN: {labels_path} missing; skipping {run_dir.name}",
              file=sys.stderr)
        return {}
    labels = json.loads(labels_path.read_text())
    excl = excluded_frames(run_dir)
    objects_dir = run_dir / "objects"

    per_frame: dict[str, list[tuple[tuple[float, float, float, float], str]]] = {}
    for tid_str, true_class in labels.items():
        track_path = objects_dir / f"track_{tid_str}.csv"
        if not track_path.exists():
            print(f"WARN: {track_path} missing for labeled T{tid_str} "
                  f"in {run_dir.name}", file=sys.stderr)
            continue
        with track_path.open() as f:
            for row in csv.DictReader(f):
                frame = row["image"]
                if frame in excl:
                    continue
                x1, y1, x2, y2 = clamp_bbox(
                    row["x1"], row["y1"], row["x2"], row["y2"], 640, 480
                )
                if x2 <= x1 or y2 <= y1:
                    print(f"WARN: degenerate bbox {(x1, y1, x2, y2)} "
                          f"in {run_dir.name}/T{tid_str}/{frame} — dropped",
                          file=sys.stderr)
                    continue
                per_frame.setdefault(frame, []).append(
                    ((x1, y1, x2, y2), true_class)
                )
    return per_frame


def build_split(batch_dir: Path, run_names: list[str], out_dir: Path,
                symlink: bool):
    """Build one split (train or val). Returns counter of per-class
    instance totals + frame count."""
    out_imgs = out_dir / "images"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "annotations.jsonl"

    # Wipe stale outputs so the build is idempotent.
    for p in out_imgs.glob("*.jpg"):
        p.unlink()
    if out_jsonl.exists():
        out_jsonl.unlink()

    cat_counter: Counter[str] = Counter()
    n_dropped_frames_no_inst = 0
    n_frames_written = 0

    idx = 0
    with out_jsonl.open("w") as jsonl:
        for run_name in run_names:
            run_dir = batch_dir / run_name
            if not run_dir.is_dir():
                print(f"WARN: {run_dir} missing; skipping",
                      file=sys.stderr)
                continue
            per_frame = collect_run_instances(run_dir)
            for frame in sorted(per_frame):
                instances = per_frame[frame]
                if not instances:
                    n_dropped_frames_no_inst += 1
                    continue
                src_img = run_dir / "images" / frame
                if not src_img.exists():
                    print(f"WARN: source image {src_img} missing",
                          file=sys.stderr)
                    continue
                dst_name = f"{idx:06d}.jpg"
                dst_img = out_imgs / dst_name
                if symlink:
                    dst_img.symlink_to(src_img.resolve())
                else:
                    shutil.copy2(src_img, dst_img)

                row = {
                    "filename": dst_name,
                    "height": 480,
                    "width": 640,
                    "source_run": run_name,
                    "source_frame": frame,
                    "detection": {
                        "instances": [
                            {
                                "bbox": [round(b[0], 1), round(b[1], 1),
                                         round(b[2], 1), round(b[3], 1)],
                                "label": CAT_ID[cat],
                                "category": cat,
                            }
                            for (b, cat) in instances
                        ],
                    },
                }
                jsonl.write(json.dumps(row) + "\n")
                for _, cat in instances:
                    cat_counter[cat] += 1
                idx += 1
                n_frames_written += 1

    return {
        "frames_written": n_frames_written,
        "instances_by_class": dict(cat_counter),
        "dropped_empty": n_dropped_frames_no_inst,
    }


def write_categories(out_root: Path):
    (out_root / "categories.json").write_text(
        json.dumps(CATEGORIES, indent=2) + "\n"
    )


def write_readme(out_root: Path, batch_dir: Path, train_runs, val_runs,
                 train_stats, val_stats):
    lines = []
    lines.append("# perception_dataset_260526")
    lines.append("")
    lines.append("Grounding-DINO finetuning dataset built from the curated")
    lines.append(f"`{batch_dir}` capture batch.")
    lines.append("")
    lines.append("Format: **ODVG-detection JSONL**. Each line is one image's")
    lines.append("`{filename, height, width, source_run, source_frame, detection: {instances: [...]}}`.")
    lines.append("Bounding boxes are absolute pixel `[x1, y1, x2, y2]` on")
    lines.append("the 640x480 frames.")
    lines.append("")
    lines.append("## Classes")
    lines.append("")
    lines.append("| id | name |")
    lines.append("|---:|---|")
    for c in CATEGORIES:
        lines.append(f"| {c['id']} | {c['name']} |")
    lines.append("")
    lines.append("Inference prompt to use after finetuning:")
    lines.append("`\"transparent. metal. cardboard.\"`")
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    lines.append("| split | runs | frames | t / m / c instances |")
    lines.append("|---|---|---:|---|")
    for split, runs, stats in (
        ("train", train_runs, train_stats),
        ("val", val_runs, val_stats),
    ):
        t = stats["instances_by_class"].get("transparent", 0)
        m = stats["instances_by_class"].get("metal", 0)
        c = stats["instances_by_class"].get("cardboard", 0)
        lines.append(
            f"| {split} | {runs[0]}..{runs[-1]} ({len(runs)} runs) | "
            f"{stats['frames_written']} | "
            f"{t} / {m} / {c} |"
        )
    lines.append("")
    lines.append("## Frame filter")
    lines.append("")
    lines.append("Any frame in the source runs that fell into one of the")
    lines.append("following data-quality buckets was excluded from this")
    lines.append("dataset:")
    lines.append("")
    lines.append("- detector miss on a labeled track")
    lines.append("  (`summary.csv:missed_frames`),")
    lines.append("- the frame containing a deleted erroneous detection")
    lines.append("  (`corrections.json:_delete_track_notes`),")
    lines.append("- per-detection delete inside an otherwise-valid track")
    lines.append("  (`corrections.json:delete_detections`),")
    lines.append("- whole-frame delete (`corrections.json:delete_frames`).")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("Each annotation row carries `source_run` and `source_frame`")
    lines.append("fields. To inspect a sample's origin:")
    lines.append("`<batch>/<source_run>/images/<source_frame>` and")
    lines.append("`<batch>/<source_run>/objects/` for the curation artifacts")
    lines.append("(`true_labels.json`, `corrections.json`).")
    lines.append("")
    lines.append("Rebuilt deterministically by")
    lines.append("`scripts/build_finetune_dataset.py`.")
    (out_root / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    batch_dir = Path(args.batch).resolve()

    print(f"batch: {batch_dir}", flush=True)
    print(f"output: {out_root.resolve()}", flush=True)
    print(f"train runs: {args.train}", flush=True)
    print(f"val runs: {args.val}", flush=True)
    print(f"mode: {'symlink' if args.symlink else 'copy'}", flush=True)
    print("", flush=True)

    print("=== train ===", flush=True)
    train_stats = build_split(batch_dir, args.train, out_root / "train",
                              args.symlink)
    print(f"  frames: {train_stats['frames_written']}", flush=True)
    print(f"  instances: {train_stats['instances_by_class']}", flush=True)

    print("=== val ===", flush=True)
    val_stats = build_split(batch_dir, args.val, out_root / "val",
                            args.symlink)
    print(f"  frames: {val_stats['frames_written']}", flush=True)
    print(f"  instances: {val_stats['instances_by_class']}", flush=True)

    write_categories(out_root)
    write_readme(out_root, batch_dir, args.train, args.val,
                 train_stats, val_stats)

    print("", flush=True)
    print(f"done. dataset root: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
