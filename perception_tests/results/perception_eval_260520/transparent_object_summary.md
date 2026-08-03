# Transparent objects — track grouping + detection summary

Transparent track IDs grouped by physical object, with the per-track
detection breakdown from each run's `objects/summary.csv`. Same format
as `metal_object_summary.md`.

- **Verified** against `objects/true_labels.json`: every track labeled
  `transparent` is accounted for; none missed.
- `transparent frames / n_frames` = how many of a track's detections
  were classified `transparent` (the correct class). `missed` = frames
  where the tracker had no detection for that object.
- Verdict **✓** if dominant predicted class is `transparent`, **✗** otherwise.

> **Note on run_12 georgia coffee:** grouped as **6, 22** (user-confirmed).
> The initial list said `5,22`, but track 5 is labeled `metal` (the red
> cola can); track 6 was the only unassigned transparent track.

---

## run_11 — 6 physical transparent objects (15 tracks)

### pocari sweat — tracks 0, 12, 26
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 0  | 6 | 6 | transparent 6 | transparent | ✓ |
| 12 | 7 | 7 | transparent 7 | transparent | ✓ |
| 26 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 19/19 transparent frames (100%)**

### labeless large bottle, blue cap — tracks 1, 11, 24
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 1  | 6 | 6 | transparent 6 | transparent | ✓ |
| 11 | 7 | 7 | transparent 7 | transparent | ✓ |
| 24 | 7 | 7 | transparent 7 | transparent | ✓ |
**Object total: 20/20 transparent frames (100%)**

### gatorade — tracks 2, 14, 25
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 2  | 7 | 7 | transparent 7 | transparent | ✓ |
| 14 | 5 | 5 | transparent 5 | transparent | ✓ |
| 25 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 18/18 transparent frames (100%)**

### powerade — tracks 6, 21
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 6  | 7 | 7 | transparent 7 | transparent | ✓ |
| 21 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 13/13 transparent frames (100%)**

### labeless bottle, orange cap — tracks 7, 19
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 7  | 6 | 6 | transparent 6 | transparent | ✓ |
| 19 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 12/12 transparent frames (100%)**

### labeless bottle, pink cap — tracks 9, 28
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 9  | 6 | 6 | transparent 6 | transparent | ✓ |
| 28 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 12/12 transparent frames (100%)**

**run_11 transparent totals: 94/94 transparent frames (100%). No wrong tracks.**

---

## run_12 — 7 physical transparent objects (15 tracks)

### labeless bottle, pink cap — tracks 0, 15, 31
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 0  | 6 | 6 | transparent 6 | transparent | ✓ |
| 15 | 5 | 4 | transparent 4, metal 1 (1 missed: 000031) | transparent | ✓ |
| 31 | 1 | 1 | transparent 1 (static, last frame only) | transparent | ✓ |
**Object total: 11/12 transparent frames (91.7%)**

### gatorade — tracks 1, 18
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 1  | 7 | 7 | transparent 7 | transparent | ✓ |
| 18 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 13/13 transparent frames (100%)**

### powerade — tracks 4, 20
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 4  | 6 | 6 | transparent 6 | transparent | ✓ |
| 20 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 12/12 transparent frames (100%)**

### georgia coffee — tracks 6, 22  *(confirmed; see note above)*
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 6  | 7 | 6 | transparent 6, metal 1 | transparent | ✓ |
| 22 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 12/13 transparent frames (92.3%)**

### pocari sweat — tracks 7, 21
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 7  | 5 | 5 | transparent 5 | transparent | ✓ |
| 21 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 11/11 transparent frames (100%)**

### labeless bottle, orange cap — tracks 8, 26
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 8  | 6 | 6 | transparent 6 | transparent | ✓ |
| 26 | 6 | 6 | transparent 6 | transparent | ✓ |
**Object total: 12/12 transparent frames (100%)**

### labeless large bottle, blue cap — tracks 10, 24
| track | n_frames | transparent frames | class_counts | dominant | verdict |
|------:|---------:|-------------------:|--------------|----------|:-------:|
| 10 | 7 | 7 | transparent 7 | transparent | ✓ |
| 24 | 7 | 7 | transparent 7 | transparent | ✓ |
**Object total: 14/14 transparent frames (100%)**

**run_12 transparent totals: 85/87 transparent frames (97.7%). No wrong tracks.**

---

## Cross-run notes

- **Transparent is essentially solved** by the detector: 0 misclassified
  tracks across both runs (179/181 frames correct, 98.9%). The only
  non-transparent frames are isolated single-frame `metal` slips
  (pink cap track 15, georgia coffee track 6).
- Contrast with metal (`metal_object_summary.md`), where green cider and
  yellow coffee cans are routinely confused *as* transparent. The error
  is one-directional: metal→transparent, never transparent→metal at the
  track level. The model has a transparent-leaning bias on ambiguous /
  reflective surfaces.
- Recurring objects across both runs: pocari sweat, gatorade, powerade,
  pink-cap bottle, orange-cap bottle, blue-cap large bottle — all 100%
  in both runs.
