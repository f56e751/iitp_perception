# Static runs 1–4 summary (perception_eval_260522, 2026-05-22)

Each run = 20 s static capture at 1.0 s interval (≤20 frames), single
object, evaluated with the Grounding-DINO detector. `detected_frames` <
20 means some frames produced no in-vocab detection (typically early
auto-exposure–settling frames).

## Detection counts per run

| run | detected / total | metal | transparent | cardboard | dominant | mean score (T / M / C) |
|-----|:----------------:|:-----:|:-----------:|:---------:|----------|------------------------|
| run_1 | 20 / 20 | 20 | 0 | 0 | **metal** | 0.153 / 0.578 / 0.031 |
| run_2 | 18 / 20 | 18 | 0 | 0 | **metal** | 0.118 / 0.609 / 0.013 |
| run_3 | 20 / 20 | 0 | 20 | 0 | **transparent** | 0.862 / 0.019 / 0.012 |
| run_4 | 17 / 20 | 10 | 7 | 0 | **metal (unstable)** | 0.339 / 0.385 / 0.030 |

(T = transparent, M = metal, C = cardboard; scores averaged over detected frames.)

## Read

- **run_1, run_2** — clean metal object. Unanimous `metal`, mean metal
  score ~0.58–0.61, transparent well below. Stable.
- **run_3** — transparent object. Unanimous `transparent` at high
  confidence (mean 0.862); metal/cardboard near zero. The clean,
  easy case.
- **run_4** — ambiguous/reflective object. Classification flips
  frame-to-frame (10 metal vs 7 transparent) with mean metal (0.385)
  and transparent (0.339) almost tied. This is the known
  **metal→transparent** ambiguity: a reflective metal surface whose
  transparent score rivals its metal score under AE/noise. No frame was
  ever mislabeled the other direction at high confidence — the error is
  one-directional, consistent with the 260520 findings.

## Files

```
summary_runs_1_4/
├── SUMMARY.md          (this file)
└── first_frames/
    ├── run_1_first.jpg (000000.jpg of each run)
    ├── run_2_first.jpg
    ├── run_3_first.jpg
    └── run_4_first.jpg
```
