"""Shared test setup: import paths + scores.csv fixtures.

Importing this module puts the repo root and scripts/ on sys.path so tests can
`import iitp_object_detector`, `import eval_detector`, `import group_by_object`,
etc. regardless of how the test runner was invoked.
"""
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Column order produced by scripts/eval_detector.py.
SCORES_HEADER = [
    "image", "box_id", "x1", "y1", "x2", "y2",
    "score_transparent", "score_metal", "score_cardboard",
    "top_score", "predicted_class",
]


def score_row(image, box_id, box, cls, scores=None):
    """Build one scores.csv row dict.

    box is (x1, y1, x2, y2). `cls` is the predicted class; its per-class score
    is forced to 0.5 (and becomes top_score) unless overridden via `scores`.
    """
    x1, y1, x2, y2 = box
    s = {"transparent": 0.10, "metal": 0.10, "cardboard": 0.10}
    if scores:
        s.update(scores)
    else:
        s[cls] = 0.50
    top = max(s["transparent"], s["metal"], s["cardboard"])
    return {
        "image": image,
        "box_id": str(box_id),
        "x1": f"{x1}", "y1": f"{y1}", "x2": f"{x2}", "y2": f"{y2}",
        "score_transparent": f'{s["transparent"]}',
        "score_metal": f'{s["metal"]}',
        "score_cardboard": f'{s["cardboard"]}',
        "top_score": f"{top}",
        "predicted_class": cls,
    }


def write_scores(path, rows):
    """Write rows (list of score_row dicts) to a scores.csv at `path`."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORES_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# Per-track CSV layout produced by group_by_track.py / apply_corrections.py.
TRACK_HEADER = ["track_id"] + SCORES_HEADER


def write_track_csv(path, track_id, rows):
    """Write a per-track CSV (track_id column + score columns) at `path`."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACK_HEADER)
        w.writeheader()
        for r in rows:
            out = {"track_id": track_id}
            out.update(r)
            w.writerow(out)
