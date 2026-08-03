"""Build a per-run track gallery for fast manual review.

For each track in <input-dir>/objects/summary.csv, render a horizontal
strip of cropped bbox thumbnails (one per detection) labeled with
track id, predicted dominant class, n_frames, and missed_frames. Stack
all rows vertically into <input-dir>/track_gallery.png.

Lets the reviewer scan every tracked object at once and spot
misclassifications or fragmented/merged tracks without flipping
through individual annotated_tracks/ frames.
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

PALETTE = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0),
    (255, 0, 255), (0, 128, 255), (128, 0, 255), (128, 255, 0), (255, 128, 128),
]
CLASS_ABBR = {"transparent": "t", "metal": "m", "cardboard": "c"}
CLASS_COLOR = {
    "transparent": (255, 200, 100),
    "metal":       (220, 220, 220),
    "cardboard":   (100, 180, 230),
}
RED = (0, 0, 255)
GREY_BG = (60, 60, 60)
FADED_BG = (30, 30, 30)
MISS_BG = (40, 40, 40)
MISS_FG = (120, 120, 200)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i", "--input-dir",
        default=f"perception_tests/results/perception_eval_{date.today():%y%m%d}",
    )
    p.add_argument("--thumb-size", type=int, default=100)
    p.add_argument("--caption-height", type=int, default=30)
    p.add_argument("--label-width", type=int, default=260)
    p.add_argument("--output", default=None,
                   help="Output path (default <input-dir>/track_gallery.png).")
    p.add_argument("--no-auto-misses", action="store_true",
                   help="Skip auto-recording detected miss frames into "
                        "corrections.json.")
    return p.parse_args()


def letterbox_crop(img, x1, y1, x2, y2, size):
    h_full, w_full = img.shape[:2]
    x1i = max(0, int(x1)); y1i = max(0, int(y1))
    x2i = min(w_full, int(x2)); y2i = min(h_full, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = img[y1i:y2i, x1i:x2i]
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def render_label(tid, dominant, n_frames, missed, kind, bg, color,
                 width, height):
    label = np.full((height, width, 3), bg, dtype=np.uint8)
    label[:, :8] = color
    cv2.putText(label, f"T{tid}", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    if kind == "static":
        cv2.putText(label, "static", (width - 80, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)
    dom_color = CLASS_COLOR.get(dominant, WHITE)
    cv2.putText(label, dominant.upper(), (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, dom_color, 2, cv2.LINE_AA)
    cv2.putText(label, f"n={n_frames}", (16, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    miss_disp = missed if missed else "-"
    if len(miss_disp) > 30:
        miss_disp = miss_disp[:27] + "..."
    cv2.putText(label, f"miss={miss_disp}", (16, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    return label


def render_thumb_block(img, row, dominant, color, thumb_size,
                       caption_height, bg):
    thumb = letterbox_crop(img,
                           float(row["x1"]), float(row["y1"]),
                           float(row["x2"]), float(row["y2"]),
                           thumb_size)
    flipped = row["predicted_class"] != dominant
    border_color = RED if flipped else color
    border_thick = 3 if flipped else 2
    cv2.rectangle(thumb, (0, 0),
                  (thumb_size - 1, thumb_size - 1),
                  border_color, border_thick)
    cap = np.full((caption_height, thumb_size, 3), bg, dtype=np.uint8)
    frame_no = row["image"].replace(".jpg", "").lstrip("0") or "0"
    cls = CLASS_ABBR.get(row["predicted_class"], "?")
    cv2.putText(cap, f"{frame_no} {cls}", (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                RED if flipped else WHITE, 1, cv2.LINE_AA)
    return np.vstack([thumb, cap])


def render_miss_block(frame_idx, thumb_size, caption_height, bg):
    thumb = np.full((thumb_size, thumb_size, 3), MISS_BG, dtype=np.uint8)
    cv2.rectangle(thumb, (0, 0), (thumb_size - 1, thumb_size - 1),
                  MISS_FG, 1)
    (tw, th), _ = cv2.getTextSize("MISS", cv2.FONT_HERSHEY_SIMPLEX,
                                  0.55, 2)
    cv2.putText(thumb, "MISS",
                ((thumb_size - tw) // 2, (thumb_size + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, MISS_FG, 2, cv2.LINE_AA)
    cap = np.full((caption_height, thumb_size, 3), bg, dtype=np.uint8)
    cv2.putText(cap, str(frame_idx), (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MISS_FG, 1, cv2.LINE_AA)
    return np.vstack([thumb, cap])


def frame_idx_from_name(image_name):
    return int(image_name.replace(".jpg", ""))


def collect_track_misses(summary, track_rows_by_id):
    """Return {tid: [frame_filename, ...]} of intra-track detection gaps."""
    misses: dict[int, list[str]] = {}
    for srow in summary:
        rows = track_rows_by_id.get(srow["track_id"], [])
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda r: r["image"])
        present = {frame_idx_from_name(r["image"]) for r in sorted_rows}
        first_idx = min(present)
        last_idx = max(present)
        for idx in range(first_idx, last_idx + 1):
            if idx in present:
                continue
            misses.setdefault(int(srow["track_id"]), []).append(
                f"{idx:06d}.jpg"
            )
    return misses


def update_corrections_with_misses(corrections_path, misses):
    """Append any newly-detected misses to corrections.json. Each tracked
    object gets a merge_tracks entry with empty `sources` (so the entry
    only carries detection_failures). Returns count of new misses added."""
    if corrections_path.exists():
        corrections = json.loads(corrections_path.read_text())
    else:
        corrections = {}
    merges = corrections.get("merge_tracks", [])
    by_target = {m["target"]: m for m in merges}
    added = 0
    for tid, miss_frames in sorted(misses.items()):
        entry = by_target.get(tid)
        if entry is None:
            entry = {
                "target": tid,
                "sources": [],
                "detection_failures": [],
                "reason": (
                    f"Auto-detected: detector miss(es) within T{tid}'s "
                    f"frame range."
                ),
            }
            merges.append(entry)
            by_target[tid] = entry
        existing = set(entry.get("detection_failures", []))
        entry.setdefault("detection_failures", [])
        for fname in miss_frames:
            if fname in existing:
                continue
            entry["detection_failures"].append(fname)
            existing.add(fname)
            added += 1
    if added == 0:
        return 0
    corrections["merge_tracks"] = merges
    corrections_path.write_text(json.dumps(corrections, indent=2) + "\n")
    return added


def render_row(summary_row, track_rows, images_dir,
               thumb_size, caption_height, label_width):
    tid = int(summary_row["track_id"])
    color = PALETTE[tid % len(PALETTE)]
    dominant = summary_row["dominant_class"]
    n_frames = summary_row["n_frames"]
    missed = summary_row.get("missed_frames", "") or ""
    kind = summary_row["kind"]
    bg = FADED_BG if kind == "static" else GREY_BG
    height = thumb_size + caption_height

    label = render_label(tid, dominant, n_frames, missed, kind, bg,
                         color, label_width, height)

    sorted_rows = sorted(track_rows, key=lambda r: r["image"])
    blocks = [label]
    if not sorted_rows:
        return np.hstack(blocks)

    row_by_idx = {frame_idx_from_name(r["image"]): r for r in sorted_rows}
    first_idx = frame_idx_from_name(sorted_rows[0]["image"])
    last_idx = frame_idx_from_name(sorted_rows[-1]["image"])

    for idx in range(first_idx, last_idx + 1):
        row = row_by_idx.get(idx)
        if row is None:
            blocks.append(render_miss_block(idx, thumb_size,
                                            caption_height, bg))
            continue
        img_path = images_dir / row["image"]
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            blocks.append(np.full((height, thumb_size, 3), BLACK, dtype=np.uint8))
            continue
        blocks.append(render_thumb_block(img, row, dominant, color,
                                         thumb_size, caption_height, bg))
    return np.hstack(blocks)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    objects_dir = input_dir / "objects"
    images_dir = input_dir / "images"
    summary_path = objects_dir / "summary.csv"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 1

    with summary_path.open() as f:
        summary = list(csv.DictReader(f))

    summary.sort(key=lambda r: (r["kind"] == "static", int(r["track_id"])))

    track_rows_by_id = {}
    for srow in summary:
        tid_str = srow["track_id"]
        track_path = objects_dir / f"track_{tid_str}.csv"
        if not track_path.exists():
            track_rows_by_id[tid_str] = []
            continue
        with track_path.open() as f:
            track_rows_by_id[tid_str] = list(csv.DictReader(f))

    def strip_len(rows):
        if not rows:
            return 0
        idxs = [frame_idx_from_name(r["image"]) for r in rows]
        return max(idxs) - min(idxs) + 1

    max_thumbs = max(
        (strip_len(track_rows_by_id[r["track_id"]]) for r in summary),
        default=0,
    )
    width = args.label_width + max_thumbs * args.thumb_size
    row_h = args.thumb_size + args.caption_height

    rendered_rows = []
    for srow in summary:
        track_rows = track_rows_by_id[srow["track_id"]]
        row_img = render_row(srow, track_rows, images_dir,
                             args.thumb_size, args.caption_height,
                             args.label_width)
        if row_img.shape[1] < width:
            bg = FADED_BG if srow["kind"] == "static" else GREY_BG
            pad = np.full((row_h, width - row_img.shape[1], 3), bg, dtype=np.uint8)
            row_img = np.hstack([row_img, pad])
        rendered_rows.append(row_img)
        rendered_rows.append(np.full((4, width, 3), BLACK, dtype=np.uint8))

    if not rendered_rows:
        print("no tracks to render", file=sys.stderr)
        return 1

    gallery = np.vstack(rendered_rows[:-1])
    out_path = Path(args.output) if args.output else input_dir / "track_gallery.png"
    cv2.imwrite(str(out_path), gallery)
    print(f"wrote {out_path} "
          f"({gallery.shape[1]}x{gallery.shape[0]} px, {len(summary)} tracks)",
          flush=True)

    if not args.no_auto_misses:
        misses = collect_track_misses(summary, track_rows_by_id)
        if misses:
            corrections_path = objects_dir / "corrections.json"
            added = update_corrections_with_misses(corrections_path, misses)
            if added:
                print(f"recorded {added} new miss frame(s) into "
                      f"{corrections_path}; re-run "
                      f"scripts/apply_corrections.py -i {input_dir} "
                      f"to apply.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
