# Metal objects — track grouping + detection summary

Metal track IDs grouped by physical object, with the per-track
detection breakdown from each run's `objects/summary.csv`.

- **Verified** against `objects/true_labels.json`: every track labeled
  `metal` is accounted for; none missed.
- `metal frames / n_frames` = how many of a track's detections were
  classified `metal` (the correct class). `missed` = frames where the
  tracker had no detection for that object.
- A track's verdict is **✓** if its dominant predicted class is `metal`,
  **✗** if it was dominated by a wrong class.

---

## run_11 — 5 physical metal objects (11 tracks)

### yellow beer can — tracks 3, 13, 23
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 3  | 6 | 5 | metal 5, transparent 1 | metal | ✓ |
| 13 | 6 | 6 | metal 6 | metal | ✓ |
| 23 | 6 | 3 | metal 3, cardboard 3 | metal (tie) | ✓ |
**Object total: 14/18 metal frames (77.8%)**

### green cider can — tracks 4, 22
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 4  | 6 | 4 | metal 4, transparent 2 | metal | ✓ |
| 22 | 6 | 2 | transparent 4, metal 2 | transparent | ✗ |
**Object total: 6/12 metal frames (50.0%)**

### yellow coffee can — tracks 5, 15
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 5  | 5 | 1 | transparent 4, metal 1 | transparent | ✗ |
| 15 | 5 | 4 | transparent 1, metal 4 (1 missed: 000034) | metal | ✓ |
**Object total: 5/10 metal frames (50.0%)**

### black cola can — tracks 8, 17
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 8  | 6 | 6 | metal 6 | metal | ✓ |
| 17 | 5 | 5 | metal 5 | metal | ✓ |
**Object total: 11/11 metal frames (100%)**

### red cola can — tracks 10, 27
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 10 | 6 | 6 | metal 6 | metal | ✓ |
| 27 | 6 | 6 | metal 6 | metal | ✓ |
**Object total: 12/12 metal frames (100%)**

**run_11 metal totals: 48/63 metal frames (76.2%). Wrong tracks: 22, 5.**

---

## run_12 — 6 physical metal objects (14 tracks)

### black cola can — tracks 2, 14, 28
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 2  | 6 | 6 | metal 6 | metal | ✓ |
| 14 | 6 | 6 | metal 6 | metal | ✓ |
| 28 | 4 | 4 | metal 4 | metal | ✓ |
**Object total: 16/16 metal frames (100%)**

### blue cola can — tracks 3, 17
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 3  | 6 | 6 | metal 6 | metal | ✓ |
| 17 | 6 | 6 | metal 6 | metal | ✓ |
**Object total: 12/12 metal frames (100%)**

### red cola can — tracks 5, 16, 30
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 5  | 6 | 5 | metal 5, transparent 1 | metal | ✓ |
| 16 | 6 | 6 | metal 6 | metal | ✓ |
| 30 | 2 | 2 | metal 2 | metal | ✓ |
**Object total: 13/14 metal frames (92.9%)**

### yellow beer can — tracks 9, 25
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 9  | 6 | 5 | metal 5, cardboard 1 | metal | ✓ |
| 25 | 5 | 5 | metal 5 | metal | ✓ |
**Object total: 10/11 metal frames (90.9%)**

### green cider can — tracks 11, 29
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 11 | 6 | 2 | transparent 4, metal 2 | transparent | ✗ |
| 29 | 4 | 2 | metal 2, transparent 2 | metal (tie) | ✓ |
**Object total: 4/10 metal frames (40.0%)**

### yellow coffee can — tracks 12, 23
| track | n_frames | metal frames | class_counts | dominant | verdict |
|------:|---------:|-------------:|--------------|----------|:-------:|
| 12 | 3 | 1 | transparent 2, metal 1 (2 missed: 000022, 000023) | transparent | ✗ |
| 23 | 3 | 0 | transparent 3 (2 missed: 000042, 000045) | transparent | ✗ |
**Object total: 1/6 metal frames (16.7%)**

**run_12 metal totals: 56/69 metal frames (81.2%). Wrong tracks: 11, 12, 23.**

---

## Cross-run notes

- **black / red / blue cola cans** are near-perfect (90–100%) in both runs.
- **green cider can** is the weakest in both runs (run_11 50%, run_12 40%)
  — consistently confused with `transparent`.
- **yellow coffee can** swings hard by run/pose: run_11 50%, run_12 16.7%.
- Recurring objects across runs (same physical can, different label IDs):
  green cider, yellow coffee, yellow beer, red cola — useful for studying
  pose/orientation sensitivity, since the same object gets different
  results depending on how it sat on the conveyor.
