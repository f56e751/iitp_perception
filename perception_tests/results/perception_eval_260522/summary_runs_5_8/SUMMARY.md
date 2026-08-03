# Static runs 5–8 summary (perception_eval_260522, 2026-05-23)

Continuation of the static experiment series started 2026-05-22 (see
`summary_runs_1_4/`). Each run = 20 s static capture at 1.0 s interval,
single object, evaluated with the Grounding-DINO detector.

`captured_frames` < 20 means the auto-stop fired before the 20th save
(cold-camera warmup eats into the duration window; see SUMMARY note
in `summary_runs_1_4/`). `detected_frames` < `captured_frames` means
some frames produced no in-vocab detection (typically the AE-settling
first frame).

`run_5` has an additional one-frame transparent detection (`object_1`,
1 frame at the right frame edge ≈(611, 237)) that is a background /
edge artifact, not the main object. **It is excluded from the table
and means below** — all counts and scores below are computed from
`object_0` (the main object) only.

## Detection counts per run

| run | captured / detected | metal | transparent | cardboard | dominant | mean score (T / M / C) | object_0 center |
|-----|:-------------------:|:-----:|:-----------:|:---------:|----------|------------------------|-----------------|
| run_5 | 19 / 18 | 18 | 0 | 0 | **metal** | 0.087 / 0.701 / 0.029 | (330, 217) |
| run_6 | 20 / 17 | 0 | 17 | 0 | **transparent** | 0.542 / 0.149 / 0.072 | (331, 203) |
| run_7 | 20 / 20 | 0 | 20 | 0 | **transparent** | 0.736 / 0.075 / 0.031 | (234, 353) |
| run_8 | 19 / 19 | 19 | 0 | 0 | **metal** | 0.113 / 0.639 / 0.017 | (418, 341) |

(T = transparent, M = metal, C = cardboard; scores averaged over the
main object's detected frames.)

## Read

- **run_5** — metal object, clean main-object classification (18/18
  metal in `object_0`). Mean metal 0.701, transparent 0.087. A
  single-frame edge-region transparent detection (`object_1`) was
  excluded as a background artifact.
- **run_6** — transparent object. All 17 detections classified
  `transparent`; 3 frames had no in-vocab detection (`000005`,
  `000011`, `000013`, `000019` — intermittent dropouts on a
  transparent target, consistent with transparent being harder to
  lock onto frame-to-frame). Mean transparent 0.542 — lower than
  run_7 and run_3, suggesting a harder-to-see transparent instance.
- **run_7** — transparent object, the cleanest of the four: 20/20
  detected and unanimous `transparent` at mean 0.736. Comparable in
  quality to run_3 (mean 0.862) from the 1–4 batch.
- **run_8** — metal object. 19/19 detected, unanimous `metal`, mean
  0.639. Comparable in quality to run_1 / run_2 from the 1–4 batch.

## Run_6 repeats (run_6_2, run_6_3) — same physical setup

While probing detector stability we re-captured the same transparent
setup as `run_6` two more times. `object_0` is the same main object at
≈(331–341, 203–236); each run also produced some edge / wall background
detections (`object_1`, `object_2` in `run_6_2`) which are **excluded
from the table below**.

| run | captured / detected | metal | transparent | cardboard | dominant | mean score (T / M / C) | object_0 center |
|-----|:-------------------:|:-----:|:-----------:|:---------:|----------|------------------------|-----------------|
| run_6   | 20 / 17 | 0  | 17 | 0 | **transparent**       | 0.542 / 0.149 / 0.072 | (331, 203) |
| run_6_2 | 20 / 18 | 12 | 6  | 0 | **metal (unstable)**  | 0.316 / 0.391 / 0.063 | (339, 206) |
| run_6_3 | 20 / 20 | 0  | 20 | 0 | **transparent**       | 0.841 / 0.020 / 0.017 | (341, 236) |

### Read

- **Big run-to-run variance for the same physical object.** Three
  back-to-back captures span unanimous transparent at high confidence
  (`run_6_3`, mean T=0.841) → unanimous transparent at moderate
  confidence (`run_6`, mean T=0.542) → mixed metal/transparent with
  near-tied mean scores (`run_6_2`, T=0.316 vs M=0.391, *dominant
  metal*). Same object, same table position, ~minutes apart.
- **`run_6_2` flipped to metal.** 12/18 frames classified `metal`,
  mean metal (0.391) edges out mean transparent (0.316). Mirrors the
  reflective-object pattern from `run_4` and from the 260520 metal-can
  data: when the surface has any reflectivity, the transparent score
  collapses and the detector swings the other way. This is the same
  one-directional ambiguity ([[project_metal_transparent_bias]]),
  appearing here on what should have been a clean transparent run.
- **`run_6_3` recovered fully.** Same object, minutes later: 20/20
  transparent at mean 0.841 — the cleanest transparent run of the
  whole batch. Implies the variance is camera-side (lighting / AE /
  glare on the surface) rather than detector-side, consistent with
  [[project_inference_determinism]].
- **Edge artifacts** (`object_1`, `object_2` in `run_6_2`, both
  transparent at the right frame edge ≈(594–615, 267)) are persistent
  background detections, not the main object.

## Files

```
summary_runs_5_8/
├── SUMMARY.md          (this file)
└── first_frames/
    ├── run_5_first.jpg (000000.jpg of each run)
    ├── run_6_first.jpg
    ├── run_6_2_first.jpg
    ├── run_6_3_first.jpg
    ├── run_7_first.jpg
    └── run_8_first.jpg
```
