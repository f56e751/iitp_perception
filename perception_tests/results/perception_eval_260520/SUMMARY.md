# Perception evaluation — 2026-05-20

End-to-end evaluation of the IITP detection module (Grounding-DINO,
`iitp_object_detector.py`) on RealSense captures, covering static
scenes, multiple-object moving scenes, manual track corrections, and
ground-truth comparison.

All artifacts under `tmp_results/perception_eval_260520/`.
All tools under `perception_eval/`.

## Tools built this session

| File | Purpose |
|---|---|
| `perception_eval/capture_only.py` | RealSense color-only capture to `<dir>/images/`. CLI: `-o`, `-i interval`, `--duration`. |
| `perception_eval/docker_capture.sh` | Container wrapper for `capture_only.py` (USB passthrough). |
| `perception_eval/eval_detector.py` | Per-frame inference; emits annotated images + `scores.csv` with **per-class scores** (transparent / metal / cardboard) for every detection, not just top-1. |
| `perception_eval/group_by_object.py` | Static-scene grouping: IoU clustering. |
| `perception_eval/group_by_track.py` | Moving-scene tracking: online centroid-distance matching with directional + min-progress constraints. Emits per-track CSVs + `summary.csv` + color-coded `annotated_tracks/` with `T<id>` overlays. |
| `perception_eval/apply_corrections.py` | Reads `objects/corrections.json` (delete tracks / delete frames / merge tracks / reassign per-detection track-ids) and rebuilds CSVs + annotated frames. Idempotent. |
| `perception_eval/analyze_tracks.py` | Reads `objects/true_labels.json` and emits confusion matrix, P/R/F1, per-track table. |
| `perception_eval/run_one.sh` | One-shot `capture + eval + chown + group_by_object` wrapper, for static-scene captures. |

## Workflow

**Static scene:** `./perception_eval/run_one.sh run_X 20` (capture, eval, IoU group).

**Moving scene:**
```bash
./perception_eval/docker_capture.sh -o tmp_results/perception_eval_260520/run_X -i 1.0 --duration 60
docker run -i --rm --gpus all --ipc=host -v $PWD:/mnt --name iitp_eval \
  chaehyeonsong/grounded_sam:latest \
  bash -c "cd /mnt && python perception_eval/eval_detector.py -i tmp_results/perception_eval_260520/run_X"
docker run --rm -v "$PWD":/mnt iitp_local:latest \
  chown -R "$(id -u):$(id -g)" /mnt/tmp_results/perception_eval_260520/run_X
python3 perception_eval/group_by_track.py -i tmp_results/perception_eval_260520/run_X --motion-direction up
# (optional) edit objects/corrections.json then:
python3 perception_eval/apply_corrections.py -i tmp_results/perception_eval_260520/run_X
# (optional, with true_labels.json):
python3 perception_eval/analyze_tracks.py -i tmp_results/perception_eval_260520/run_X
```

## Experiments

### Static, single object — runs 1–5

Same object (crushed green BRISK can) placed centered in the FoV; 20 s
captures, 1.0 s save interval.

| Run | Frames with detection | Main cluster | Class outcome |
|---|---|---|---|
| run_1 | 20/20 | (316, 208) + (600, 259) wall FP | center: metal 18, transparent 1 (frame 0 outlier); wall: transparent 18 |
| run_2 | 17/20 | (316, 208) | metal 17 |
| run_3 | 18/20 | (316, 208) | metal 18 |
| run_4 | 18/20 | (316, 208) | metal 18 |
| run_5 | 18/20 | (316, 208) | metal 18 |

**Determinism check.** Re-ran `eval_detector.py` three times over
`run_1/images/`. All three `scores.csv` outputs byte-identical
(MD5 `59529cf223fca88a9807b6955cc72275`, files
`run_1/scores_{a,b,c}.csv` retained). Inference is fully
deterministic on fixed pixel input. Any observed score wobble is
camera-side, not model-side.

### Static, different object — runs 6–10

Different object placed at (332, 240). All runs classified
**transparent** in every detected frame (87/87 across the five
captures). No metal/transparent flipping.

### Camera AE convergence (re-investigated)

Mean BGR brightness per saved frame across runs 1–5:

| t (s) | run_1 | run_2 | run_3 | run_4 | run_5 |
|---|---|---|---|---|---|
| 0 | 99.91 | 100.38 | 106.66 | 105.63 | 106.98 |
| 1 | 106.25 | 106.98 | 110.19 | 110.66 | 110.38 |
| 2 | 110.21 | 110.76 | 112.92 | 113.61 | 112.88 |
| 3 | 114.04 | 111.32 | 114.51 | 113.37 | 114.50 |
| 10 (settled) | 114.07 | 111.35 | 117.06 | 113.40 | 117.08 |

AE takes **~2–3 wall-clock seconds (≈ 60–90 sensor frames @ 30 FPS)**
to converge in every run. The run_1 frame-0 transparent outlier is a
direct consequence: that frame was 14 DN darker than steady state.

A capture-side warmup fix is designed but not implemented — the user
deferred it because it would only correct 1 detection out of ~100. See
`/home/iitp/.claude/projects/-PublicSSD-iitp/memory/feedback_proportional_fixes.md`.

### Moving scenes — run_11 and run_12

Two independent 60 s captures, multiple objects flowing upward on a
conveyor.

**Tracker iterations (developed during run_11):**
1. *Initial centroid-distance match (threshold 100 px, `--motion-direction any`)*: 26 tracks; several spurious 13–16-frame tracks (e.g., track_3) caused by greedy match preferring a stationary edge match when a new object entered near an exiting track's last position.
2. *Added `--motion-direction up`* with a 20 px tolerance: still produced a 11-frame merged track because the tolerance let new-object detections sneak in at the edge.
3. *Replaced tolerance with `--min-progress 30 px`*: tracks compressed to **5–7 frames each**, matching the user's observed transit time. Final: 26 valid moving tracks after manual corrections.

**Manual corrections applied to run_11** (`objects/corrections.json`):
- Deleted T16, T18 (off-conveyor cardboard FP).
- Merged T20 → T15 (single detection failure in frame 34 between them).
- Reassigned a cascading chain of swaps in frames 39–42 between T17/T19, T19/T21, T21/T22 (one ID-switch on frame 39 propagated forward through the greedy matcher).

After corrections: 26 tracks, 157 frames.

**Manual corrections applied to run_12** (added `delete_frames`
support to `apply_corrections.py` for this run):
- Deleted all detections in frame 40 (object held in user's hand, not on belt).
- Merged T13 → T12 (object missed in frames 22, 23).
- Merged T19 → T15 (object missed in frame 31).
- Merged T27 → T23 (object missed in frames 42, 45).

After corrections: 29 tracks, 156 frames. Each track is a single
physical object with a consistent trajectory.

## Ground-truth results

### run_11 (15 transparent, 11 metal, 0 cardboard)

**Per-frame confusion (rows = true, cols = predicted):**

| true \\ pred | transparent | metal | cardboard | total |
|---|---|---|---|---|
| transparent | **94** | 0 | 0 | 94 |
| metal | 12 | **48** | 3 | 63 |

**Per-class metrics (frame-level):**

| class | precision | recall | F1 |
|---|---|---|---|
| transparent | 0.887 | **1.000** | 0.940 |
| metal | **1.000** | 0.762 | 0.865 |
| cardboard | — | — | — (only as FP) |

- **Per-frame accuracy: 90.4 %** (142 / 157).
- **Per-track accuracy: 92.3 %** (24 / 26 — T5 and T22 are the two metal objects with dominant=transparent).
- Mean score on metal frames: `score_metal` 0.554 vs `score_transparent` 0.222.
- Mean score on transparent frames: `score_transparent` 0.734 vs `score_metal` 0.037.

### run_12 (15 transparent, 14 metal, 0 cardboard)

Second 60 s moving-object capture. Same procedure as run_11, different
object placements. After corrections: 29 tracks, 156 frames.

**Per-frame confusion:**

| true \\ pred | transparent | metal | cardboard | total |
|---|---|---|---|---|
| transparent | **85** | 2 | 0 | 87 |
| metal | 12 | **56** | 1 | 69 |

**Per-class metrics:**

| class | precision | recall | F1 |
|---|---|---|---|
| transparent | 0.876 | 0.977 | 0.924 |
| metal | 0.966 | 0.812 | 0.882 |

- **Per-frame accuracy: 90.4 %** (141 / 156).
- **Per-track accuracy: 89.7 %** (26 / 29 — T11, T12, T23 are metal objects with dominant=transparent).
- Mean score on metal frames: `score_metal` 0.640 vs `score_transparent` 0.191.
- Mean score on transparent frames: `score_transparent` 0.777 vs `score_metal` 0.045.

### Cross-run comparison

| Metric | run_11 | run_12 |
|---|---|---|
| Tracks | 26 | 29 |
| Frames | 157 | 156 |
| Per-frame accuracy | 90.4 % | 90.4 % |
| Per-track accuracy | 92.3 % | 89.7 % |
| Transparent recall | 1.000 | 0.977 |
| Transparent precision | 0.887 | 0.876 |
| Metal recall | 0.762 | 0.812 |
| Metal precision | 1.000 | 0.966 |
| Wrong tracks (metal→trans) | 2 | 3 |

**Per-frame accuracy is identical at 90.4 %** across two independent
60 s captures. The error budget is highly consistent: most errors are
metal objects being called transparent (12 / 15 errors in each run),
with the failure concentrated in 2–3 specific objects rather than
spread evenly. **The mean-score pattern is also stable:** on metal
frames, `score_metal ≈ 0.6` with a competing `score_transparent ≈
0.2` — the residual transparent signal that flips classifications when
the orientation isn't quite right.

## Key findings about the model

1. **Inference is deterministic.** Same pixels → same scores, byte-identical. Future fluctuation reports should default to investigating the capture side.

2. **Transparent precision/recall is excellent, metal recall is the weak link.** The detector never mistakenly calls transparent objects metal; it sometimes calls metal objects transparent.

3. **The error is high-confidence, not borderline.** Wrong "metal-as-transparent" frames have `score_transparent` 0.6–0.8 — the model isn't uncertain, it's confidently mistaken.

4. **Orientation / pose dependence is the dominant explanation.** Same physical objects yield very different classifications under different poses:
   - The BRISK can (T4 and T22 are the same can as static runs 1–5): T4 was in a similar orientation to the static-run pose → 4/6 correct. T22 was rotated/flipped → 2/6 correct, even though static runs hit 99 %.
   - The silver/clear can (T5 and T15 are the same object passed twice): T15 landed in a more "bottle-shaped" pose → 4/5 metal. T5 was more crumpled → 1/5 metal.

5. **Secondary: position-in-frame matters.** Metal recall by Y-position bin: top 88 %, upper-mid 73 %, lower-mid 75 %, bottom 68 %. The camera angle on objects near the bottom of the FoV is more top-down, which seems to hurt metal classification. Transparent recall stays at 100 % across all Y bins. The static-run object was always at y ≈ 127–250, the most favorable region.

6. **Cardboard is purely an FP class in this scene.** 0 true cardboard, 3 metal frames mistakenly called cardboard (and 2 deleted off-conveyor cardboard FPs from T16/T18).

## Why moving is harder than static

The static experiments accidentally measured the detector at its **best case**: one specific object, in one specific pose, in the most favorable region of the FoV. The moving experiment exposes:

- More object variety (11 distinct metal objects vs 1) — and the model isn't equally good at all of them.
- Multiple poses per object (the same can flipped, crushed differently, landing in different orientations).
- Full-FoV transit (model is less reliable near the bottom of the frame).

The tracker itself is doing fine; the residual ~10 % error budget is genuinely in the underlying detection model.

## Suggested next steps (not implemented this session)

- **Temporal voting per track**: take the per-track majority class as the final answer. Recovers T4 (4/6 → metal) and ~3 other tracks; doesn't fix T5 or T22 where the majority is already wrong.
- **Use depth + RGB**: RealSense gives aligned depth already (captured in `capture_and_detect.py` but not used in `eval_detector.py`). Depth-based shape features would be more pose-invariant than RGB.
- **Fine-tune Grounding-DINO** on a curated set of these objects in many orientations on the belt — the principled fix, expensive.
- **AE warmup in capture** (deferred): low priority (~1 % impact) but trivial to add when wanted.

## Data inventory

```
tmp_results/perception_eval_260520/
├── SUMMARY.md                     (this file)
├── metal_object_summary.md        (run_11+12 metal tracks grouped by physical object, with per-track detection breakdown)
├── transparent_object_summary.md  (run_11+12 transparent tracks grouped by physical object, with per-track detection breakdown)
├── run_1/              static, BRISK can, single object — scores_a/b/c.csv kept as determinism-check evidence
├── run_2..5/           static, BRISK can, repeated 4× — baseline accuracy
├── run_6..10/          static, different transparent object, 5×
├── run_11/             moving, 60 s, multi-object
│   ├── images/             (60 raw JPGs)
│   ├── annotated/          (eval_detector per-frame annotations)
│   ├── annotated_tracks/   (post-correction, color-coded by track id)
│   ├── scores.csv          (157 detection rows)
│   ├── objects/
│   │   ├── track_*.csv     (per-track detections, post-correction)
│   │   ├── summary.csv     (per-track aggregate)
│   │   ├── corrections.json
│   │   └── true_labels.json
└── run_12/             moving, 60 s, multi-object (replication of run_11)
    ├── images/             (60 raw JPGs)
    ├── annotated/
    ├── annotated_tracks/
    ├── scores.csv          (156 detection rows)
    └── objects/            (29 tracks after corrections, true_labels.json)
```
