"""Analyze tracker output against ground-truth labels.

Reads:
  <input-dir>/objects/track_*.csv        (corrected per-track detections)
  <input-dir>/objects/true_labels.json   ({track_id_str: class_name})

Produces console output:
  - Per-frame confusion matrix (true class -> predicted class)
  - Per-class precision / recall / F1
  - Per-frame and per-track accuracy
  - Mean per-class scores conditional on TRUE class (vs the per-track-dominant
    version, which is misleading when the detector is wrong for most frames)
  - Per-track summary highlighting tracks where prediction != true label
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ["transparent", "metal", "cardboard"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i", "--input-dir",
        default="tmp_results/perception_eval_260520",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    objects_dir = Path(args.input_dir) / "objects"
    labels_path = objects_dir / "true_labels.json"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.")
        return 1
    true_labels = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}

    tracks = {}
    for p in sorted(objects_dir.glob("track_*.csv")):
        tid = int(p.stem.split("_")[1])
        with p.open() as f:
            tracks[tid] = list(csv.DictReader(f))

    # Restrict to tracks with a true label.
    labeled_tids = [t for t in tracks if t in true_labels]
    missing = sorted(set(tracks) - set(true_labels))
    extra = sorted(set(true_labels) - set(tracks))
    if missing:
        print(f"WARN: tracks without ground-truth label: {missing}")
    if extra:
        print(f"WARN: labels for missing tracks: {extra}")

    # Frame-level confusion: true x predicted.
    confusion = defaultdict(lambda: defaultdict(int))
    score_sums = defaultdict(lambda: [0.0, 0.0, 0.0])  # true class -> [t,m,c] sums
    score_counts = Counter()
    track_correct = Counter()  # tracks whose dominant matches true
    track_total = Counter()

    n_frames_total = 0
    n_frames_correct = 0

    for tid in labeled_tids:
        true_cls = true_labels[tid]
        rows = tracks[tid]
        track_total[true_cls] += 1
        # Dominant class for this track:
        dom = Counter(r["predicted_class"] for r in rows).most_common(1)[0][0]
        if dom == true_cls:
            track_correct[true_cls] += 1
        for r in rows:
            pred = r["predicted_class"]
            confusion[true_cls][pred] += 1
            score_sums[true_cls][0] += float(r["score_transparent"])
            score_sums[true_cls][1] += float(r["score_metal"])
            score_sums[true_cls][2] += float(r["score_cardboard"])
            score_counts[true_cls] += 1
            n_frames_total += 1
            if pred == true_cls:
                n_frames_correct += 1

    print(f"Tracks labeled: {len(labeled_tids)}")
    print(f"Frames labeled: {n_frames_total}")
    print()

    # Object counts by true class.
    print("Objects per TRUE class:")
    for c in CLASSES:
        print(f"  {c:12s} {track_total[c]} objects")
    print()

    # Confusion matrix.
    print("Per-frame confusion (rows = TRUE, cols = predicted):")
    header_label = "true_vs_pred"
    print(f"  {header_label:14s}" + "".join(f"{c:>13s}" for c in CLASSES) + f"{'total':>10s}")
    for t in CLASSES:
        row = confusion.get(t, {})
        total = sum(row.values())
        if total == 0 and track_total[t] == 0:
            continue
        cells = "".join(f"{row.get(p, 0):>13d}" for p in CLASSES)
        print(f"  {t:14s}{cells}{total:>10d}")
    print()

    # Per-class precision/recall/F1.
    print("Per-class metrics (frame-level):")
    print(f"  {'class':12s} {'precision':>10s} {'recall':>10s} {'F1':>10s}  (TP / FN / FP)")
    for c in CLASSES:
        TP = confusion[c][c]
        FN = sum(confusion[c].values()) - TP
        FP = sum(confusion[t][c] for t in CLASSES if t != c)
        if TP + FP == 0 and TP + FN == 0:
            continue
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"  {c:12s} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}  ({TP} / {FN} / {FP})")
    print()

    # Overall accuracy.
    if n_frames_total:
        print(f"Per-frame accuracy:  {n_frames_correct}/{n_frames_total} = {n_frames_correct/n_frames_total*100:.1f}%")
    track_corr_total = sum(track_correct.values())
    track_tot_total = sum(track_total.values())
    if track_tot_total:
        print(f"Per-track accuracy:  {track_corr_total}/{track_tot_total} = {track_corr_total/track_tot_total*100:.1f}%   "
              "(track is 'correct' if its dominant class matches truth)")
    print()

    # Per-class score conditional on TRUE class.
    print("Mean per-class score, conditional on TRUE class:")
    print(f"  {'true':12s}  mean_t   mean_m   mean_c   (n_frames)")
    for c in CLASSES:
        n = score_counts[c]
        if n == 0:
            continue
        s = score_sums[c]
        print(f"  {c:12s} {s[0]/n:.3f}    {s[1]/n:.3f}    {s[2]/n:.3f}    ({n})")
    print()

    # Per-track table highlighting mismatches.
    print("Per-track results (✗ marks tracks whose dominant != true):")
    print(f"  {'tid':>3s}  {'true':12s} {'dom':12s} {'n':>2s}  {'wrong_frames':>12s}")
    for tid in sorted(labeled_tids):
        rows = tracks[tid]
        true_cls = true_labels[tid]
        counts = Counter(r["predicted_class"] for r in rows)
        dom = counts.most_common(1)[0][0]
        wrong = sum(1 for r in rows if r["predicted_class"] != true_cls)
        mark = "  ✗" if dom != true_cls else ""
        print(f"  {tid:>3d}  {true_cls:12s} {dom:12s} {len(rows):>2d}  {wrong:>12d}{mark}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
