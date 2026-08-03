"""Frame-by-frame multi-object tracker for moving scenes.

Groups detections in scores.csv into tracks by online nearest-centroid
matching across consecutive frames. Tracks that never move much are
flagged as 'static' (likely off-conveyor false positives); the rest
are 'moving' (real objects transiting the FoV).

Pairs with eval_detector.py: reads <input-dir>/scores.csv, writes
<input-dir>/objects/track_<i>.csv and <input-dir>/objects/summary.csv.

Use this for moving-object captures. For static scenes use
group_by_object.py instead.
"""

import argparse
import csv
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import cv2

# Distinct BGR colors cycled by `track_id % len(PALETTE)`.
PALETTE = [
    (0, 255, 0),     # green
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 255, 0),   # cyan
    (255, 0, 255),   # magenta
    (0, 128, 255),   # orange
    (128, 0, 255),   # purple
    (128, 255, 0),   # lime
    (255, 128, 128), # light blue
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input-dir",
        default=f"perception_tests/results/perception_eval_{date.today():%y%m%d}",
        help="Folder containing scores.csv.",
    )
    p.add_argument(
        "--distance-threshold",
        type=float,
        default=100.0,
        help="Max centroid distance (px), measured against the velocity-predicted next position, "
             "for a new detection to be appended to an existing track.",
    )
    p.add_argument(
        "--max-missed-frames",
        type=int,
        default=2,
        help="Close a track once it has been idle for more than this many consecutive frames "
             "(checked before each frame's matching, so a closed track cannot grab later detections).",
    )
    p.add_argument(
        "--min-motion",
        type=float,
        default=30.0,
        help="Tracks whose max pairwise centroid distance is below this are flagged kind=static.",
    )
    p.add_argument(
        "--motion-direction",
        choices=("up", "down", "left", "right", "any"),
        default="any",
        help="If set, only allow a track to extend with detections moving in this direction.",
    )
    p.add_argument(
        "--min-progress",
        type=float,
        default=30.0,
        help=(
            "Minimum forward motion (px) in the configured direction for a "
            "detection to extend a track. Filters out new objects appearing "
            "near an exiting track's edge position."
        ),
    )
    p.add_argument(
        "--size-weight",
        type=float,
        default=0.3,
        help="Weight on bbox-diagonal change in the match cost. Distance threshold is still "
             "applied to raw predicted-vs-actual centroid distance; size weighting only ranks "
             "the surviving candidates so that wildly-different-shaped detections lose to "
             "similar-shaped ones. Set 0 to revert to pure distance ranking.",
    )
    p.add_argument(
        "--rejoin-gap",
        type=int,
        default=4,
        help="Max frame gap for the post-pass re-association step that merges a closed track "
             "with a later-opened track whose first detection lands near the closed track's "
             "velocity-extrapolated position. Set 0 to disable.",
    )
    p.add_argument(
        "--rejoin-size-delta",
        type=float,
        default=80.0,
        help="Maximum allowed bbox-diagonal difference (px) when re-associating a closed "
             "track with a later track. Prevents stitching small cans onto large cardboard "
             "boxes just because they happened to pass through the same pixel.",
    )
    p.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip writing <input-dir>/annotated_tracks/<frame>.jpg with track-ID-colored bboxes.",
    )
    return p.parse_args()


def is_ahead(prev, new, direction: str, min_progress: float) -> bool:
    """Is `new` enough forward of `prev` along the configured motion direction?

    Image coordinates: y increases downward, x increases rightward. A
    positive `min_progress` requires the detection to have moved forward by
    at least that many pixels; small or backward motion is rejected.
    """
    if direction == "up":
        return (prev[1] - new[1]) >= min_progress
    if direction == "down":
        return (new[1] - prev[1]) >= min_progress
    if direction == "left":
        return (prev[0] - new[0]) >= min_progress
    if direction == "right":
        return (new[0] - prev[0]) >= min_progress
    return True  # "any"


def centroid(row):
    x1, y1 = float(row["x1"]), float(row["y1"])
    x2, y2 = float(row["x2"]), float(row["y2"])
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def bbox_diag(row):
    x1, y1 = float(row["x1"]), float(row["y1"])
    x2, y2 = float(row["x2"]), float(row["y2"])
    return math.hypot(x2 - x1, y2 - y1)


def predict_centroid(tr, fidx):
    """Extrapolate a track's expected centroid at frame fidx from its last
    two actually-detected centroids. Falls back to the last centroid when
    velocity cannot be estimated. Handles non-consecutive frame indices
    (gaps inside the track) by dividing the centroid delta by the frame
    delta before extrapolating."""
    cs = tr["centroids"]
    fs = tr["frame_idxs"]
    if len(cs) < 2:
        return cs[-1]
    c1, c2 = cs[-2], cs[-1]
    f1, f2 = fs[-2], fs[-1]
    if f2 == f1:
        return c2
    vx = (c2[0] - c1[0]) / (f2 - f1)
    vy = (c2[1] - c1[1]) / (f2 - f1)
    gap = fidx - f2
    return (c2[0] + vx * gap, c2[1] + vy * gap)


def render_annotated_tracks(input_dir: Path, by_frame: dict, frame_names: list) -> None:
    """For each frame, draw track-id-colored bboxes and write to annotated_tracks/."""
    images_dir = input_dir / "images"
    out_dir = input_dir / "annotated_tracks"
    out_dir.mkdir(exist_ok=True)
    for p in out_dir.glob("*.jpg"):
        p.unlink()

    n_written = 0
    for fname in frame_names:
        img_path = images_dir / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for row in by_frame[fname]:
            tid = row.get("_track_id")
            if tid is None:
                continue
            x1 = int(float(row["x1"]))
            y1 = int(float(row["y1"]))
            x2 = int(float(row["x2"]))
            y2 = int(float(row["y2"]))
            color = PALETTE[tid % len(PALETTE)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"T{tid} {row['predicted_class']} {float(row['top_score']):.2f}"
            text_y = y1 - 6 if y1 > 18 else y1 + 14
            cv2.putText(
                img, label, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(out_dir / fname), img)
        n_written += 1
    print(f"wrote {n_written} annotated frames to {out_dir}", flush=True)


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

    # Group rows by frame (image filename), preserving temporal order via sorted filenames.
    by_frame: dict[str, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(r["image"], []).append(r)
    frame_names = sorted(by_frame.keys())
    frame_index = {name: i for i, name in enumerate(frame_names)}

    # active = list of dicts: {id, rows, centroids, frame_idxs, last_centroid, last_frame_idx}
    active: list[dict] = []
    closed: list[dict] = []
    next_id = 0

    for fname in frame_names:
        fidx = frame_index[fname]
        detections = by_frame[fname]
        det_centroids = [centroid(d) for d in detections]
        det_diags = [bbox_diag(d) for d in detections]

        # Close idle tracks BEFORE this frame's matching, so a track that has
        # already exceeded max_missed_frames cannot "wake up" and absorb a
        # later detection (which was a source of class-flip cross-matches
        # in the previous version).
        still_active = []
        for tr in active:
            if fidx - tr["last_frame_idx"] > args.max_missed_frames:
                closed.append(tr)
            else:
                still_active.append(tr)
        active = still_active

        # For each (track, det) pair within the distance threshold of the
        # track's *predicted* next centroid, build a cost = distance + size
        # weight * |bbox-diagonal delta|. Distance threshold still gates;
        # size weighting only re-ranks survivors.
        candidates = []
        for ti, tr in enumerate(active):
            predicted = predict_centroid(tr, fidx)
            last_diag = bbox_diag(tr["rows"][-1])
            for di, dc in enumerate(det_centroids):
                if not is_ahead(
                    predicted, dc, args.motion_direction, args.min_progress
                ):
                    continue
                d = distance(predicted, dc)
                if d > args.distance_threshold:
                    continue
                size_delta = abs(det_diags[di] - last_diag)
                cost = d + args.size_weight * size_delta
                candidates.append((cost, ti, di))
        candidates.sort()

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _cost, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            tr = active[ti]
            tr["rows"].append(detections[di])
            tr["centroids"].append(det_centroids[di])
            tr["frame_idxs"].append(fidx)
            tr["last_centroid"] = det_centroids[di]
            tr["last_frame_idx"] = fidx
            detections[di]["_track_id"] = tr["id"]
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Unmatched detections become new tracks.
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            det["_track_id"] = next_id
            active.append(
                {
                    "id": next_id,
                    "rows": [det],
                    "centroids": [det_centroids[di]],
                    "frame_idxs": [fidx],
                    "last_centroid": det_centroids[di],
                    "last_frame_idx": fidx,
                }
            )
            next_id += 1

    # Anything still active at the end closes naturally.
    closed.extend(active)

    # Post-pass re-association. For each track (in time order), look for an
    # earlier closed track whose velocity-extrapolated position at this
    # track's first frame lands within --distance-threshold and whose last
    # bbox is similar in size to this track's first bbox. If so, merge this
    # track into the earlier one — this stitches splits caused by 1-2 missed
    # detections in the middle of an object's transit.
    if args.rejoin_gap > 0:
        closed.sort(key=lambda t: t["frame_idxs"][0])
        merged_into: dict[int, int] = {}
        for j_idx in range(len(closed)):
            later = closed[j_idx]
            if later["id"] in merged_into:
                continue
            later_first_fidx = later["frame_idxs"][0]
            later_first_centroid = later["centroids"][0]
            later_first_diag = bbox_diag(later["rows"][0])
            best = None
            for i_idx in range(j_idx):
                earlier = closed[i_idx]
                # Follow merge chain: anything merged earlier must be looked
                # up by its root.
                root_id = earlier["id"]
                while root_id in merged_into:
                    root_id = merged_into[root_id]
                root = next(t for t in closed if t["id"] == root_id)
                gap = later_first_fidx - root["last_frame_idx"]
                if gap < 1 or gap > args.rejoin_gap:
                    continue
                predicted = predict_centroid(root, later_first_fidx)
                d = distance(predicted, later_first_centroid)
                if d > args.distance_threshold:
                    continue
                size_delta = abs(later_first_diag - bbox_diag(root["rows"][-1]))
                if size_delta > args.rejoin_size_delta:
                    continue
                cost = d + args.size_weight * size_delta
                if best is None or cost < best[0]:
                    best = (cost, root)
            if best is not None:
                _, root = best
                root["rows"].extend(later["rows"])
                root["centroids"].extend(later["centroids"])
                root["frame_idxs"].extend(later["frame_idxs"])
                root["last_centroid"] = later["centroids"][-1]
                root["last_frame_idx"] = later["frame_idxs"][-1]
                for r in later["rows"]:
                    r["_track_id"] = root["id"]
                merged_into[later["id"]] = root["id"]

        closed = [tr for tr in closed if tr["id"] not in merged_into]

    closed.sort(key=lambda t: t["id"])

    # Output.
    objects_dir = input_dir / "objects"
    objects_dir.mkdir(exist_ok=True)
    for p in list(objects_dir.glob("track_*.csv")) + list(objects_dir.glob("object_*.csv")):
        p.unlink()
    summary_path = objects_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    track_header = ["track_id"] + list(header)

    with summary_path.open("w", newline="") as fsum:
        sw = csv.writer(fsum)
        sw.writerow(
            [
                "track_id",
                "kind",
                "n_frames",
                "first_frame",
                "last_frame",
                "mean_x_center",
                "mean_y_center",
                "max_motion_px",
                "dominant_class",
                "class_counts",
            ]
        )

        for tr in closed:
            tid = tr["id"]
            tr_rows = tr["rows"]
            centroids = [centroid(r) for r in tr_rows]

            # Per-track CSV. Strip internal '_'-prefixed fields (e.g. _track_id).
            with (objects_dir / f"track_{tid}.csv").open("w", newline="") as ftr:
                tw = csv.DictWriter(ftr, fieldnames=track_header)
                tw.writeheader()
                for r in tr_rows:
                    out = {"track_id": tid}
                    out.update({k: v for k, v in r.items() if not k.startswith("_")})
                    tw.writerow(out)

            # Summary stats.
            n = len(tr_rows)
            max_motion = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(centroids[i], centroids[j])
                    if d > max_motion:
                        max_motion = d
            mean_x = sum(c[0] for c in centroids) / n
            mean_y = sum(c[1] for c in centroids) / n
            classes = [r["predicted_class"] for r in tr_rows]
            counts = Counter(classes)
            dominant = counts.most_common(1)[0][0]
            kind = "moving" if max_motion >= args.min_motion else "static"

            sw.writerow(
                [
                    tid,
                    kind,
                    n,
                    tr_rows[0]["image"],
                    tr_rows[-1]["image"],
                    f"{mean_x:.1f}",
                    f"{mean_y:.1f}",
                    f"{max_motion:.1f}",
                    dominant,
                    dict(counts),
                ]
            )
            print(
                f"track_{tid}: {kind}, {n} frames, {tr_rows[0]['image']}..{tr_rows[-1]['image']}, "
                f"center≈({mean_x:.0f},{mean_y:.0f}), motion={max_motion:.0f}px, "
                f"counts={dict(counts)}",
                flush=True,
            )

    if not args.no_annotate:
        render_annotated_tracks(input_dir, by_frame, frame_names)

    return 0


if __name__ == "__main__":
    sys.exit(main())
