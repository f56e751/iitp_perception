"""Pre-fill <input-dir>/objects/true_labels.json from summary.csv.

For each moving track in `objects/summary.csv`, record
`{track_id: dominant_class}`. Skip static tracks by default (they're
usually off-conveyor false positives slated for deletion via
corrections.json); use --include-static to keep them.

Will not overwrite an existing `true_labels.json` unless --force is
passed.
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i", "--input-dir",
        default=f"perception_tests/results/perception_eval_{date.today():%y%m%d}",
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing true_labels.json.")
    p.add_argument("--include-static", action="store_true",
                   help="Also pre-fill static (likely-FP) tracks.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    objects_dir = input_dir / "objects"
    summary_path = objects_dir / "summary.csv"
    labels_path = objects_dir / "true_labels.json"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 1

    if labels_path.exists() and not args.force:
        print(f"NOTICE: {labels_path} already exists; not overwriting. "
              f"Pass --force to overwrite.", file=sys.stderr)
        return 1

    with summary_path.open() as f:
        rows = list(csv.DictReader(f))

    labels: dict[int, str] = {}
    skipped_static = 0
    for r in rows:
        if r["kind"] == "static" and not args.include_static:
            skipped_static += 1
            continue
        labels[int(r["track_id"])] = r["dominant_class"]

    sorted_labels = {str(k): labels[k] for k in sorted(labels)}
    labels_path.write_text(json.dumps(sorted_labels, indent=2) + "\n")

    msg = f"wrote {labels_path} ({len(sorted_labels)} tracks)"
    if skipped_static:
        msg += f"; skipped {skipped_static} static"
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
