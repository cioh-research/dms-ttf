# dms-ttf

Data and code for the article

> Oh, C., Kim, W., Lee, M. & Pitts, B. J. A time-to-full-warning criterion for reducing between-driver variability in drowsiness pre-warning timing. *Scientific Reports* (submitted).

The repository contains processed, de-identified eye-state time series from 46 drivers in a drowsiness-inducing driving-simulator study, and the code that replays the NHTSA staged PERCLOS thresholds and the proposed time-to-full-warning (TTF) criterion on these data. Running one script reproduces every number, table and figure in the article that is based on the data (Fig. 1 contains photographs and is not included).

## Contents

```
data/
  participants_summary.csv    aggregated characteristics of the participants (no individual records)
  eye/S<code>.csv.gz          processed eye-state time series of the 46 participants with RGB camera data
  blink_events.csv            blink events detected in the corrected eye-opening series
  preprocessing_summary.csv   per-participant counts of the preprocessing corrections
  LICENSE.md                  data licence (CC BY 4.0)
code/
  common.py                   paths, constants, data loading and the 1-s causal grid
  replay.py                   warning policies (Eqs. 5–10, Table 2) and the replay of every record
  metrics.py                  event-level outcome measures (Eq. 12)
  stats_analysis.py           paired tests, mixed-effects models; Table 1 and group comparisons (restricted data)
  level_sensitivity.py        FULL-level sensitivity (8, 10, 12%)
  parameter_sensitivity.py    one-at-a-time parameter sensitivity
  preprocessing_checks.py     frame rate, thresholds, PERCLOS recomputation, correction counts
  figures.py                  Figures 2–4
  run_all.py                  runs everything in order
results/                      outputs of run_all.py (included for reference)
figures/                      Figures 2–4 produced by run_all.py
```

## Reproducing the results

Python 3.10 or later is required. The analysis was run with Python 3.13, NumPy 2.5, pandas 3.0, SciPy 1.18, statsmodels 0.15 and Matplotlib 3.11.

```bash
pip install -r requirements.txt
python code/run_all.py
```

The full run takes about one minute on a laptop. Bootstrap confidence intervals use a fixed seed (2026), so the outputs are identical to those reported.

Individual participant characteristics are not distributed (see below). Without them, `run_all.py` reproduces everything except Table 1 and the group comparisons (Fig. 4), for which the aggregated results are provided in `results/table1_participants.json` and `results/subgroups.csv`; `figures.py` draws Fig. 4 from the latter. Authorized users who place the restricted file at `data/restricted/participants.csv` can regenerate these results as well.

| Article item | Script | Output |
|---|---|---|
| Table 1 (restricted data; aggregated output provided) | `stats_analysis.py` | `results/table1_participants.json` |
| Table 3 (descriptive columns), Table 4, Supplementary Table S2 | `metrics.py` | `results/summary_metrics.csv` |
| Table 3 (paired comparisons) | `stats_analysis.py` | `results/paired_TTF-pilot.csv` |
| Mixed-effects models, events closer to / farther from target | `stats_analysis.py` | `results/mixed_models.json` |
| Fig. 4 and participant-characteristics results (restricted data; aggregated output provided) | `stats_analysis.py`, `figures.py` | `results/subgroups.csv` |
| Supplementary Table S1 | `metrics.py` | `results/event_leads.csv` |
| Supplementary Table S4 | `level_sensitivity.py` | `results/levels_table.csv` |
| Supplementary Table S5 | `parameter_sensitivity.py` | `results/param_sens.csv` |
| Methods (frame rate, corrections, minimum passage interval) | `preprocessing_checks.py` | `results/preprocessing_checks.json` |
| Figs. 2–4 | `figures.py` | `figures/` |

Policy names in the code follow the article: `NHTSA-replay`, `NHTSA-replay-matched`, `TTF-pilot` (proposed), `TTF-pilot-slow-only`, `TTF-pilot-fast-only`, `TTF-pilot-2nd-order` and `TTF-pilot-3rd-order`. In `replay_events.csv`, `init = startup` marks the initialization sensitivity run of `TTF-pilot` (Supplementary Table S2).

## Using the TTF criterion on other data

`replay.run_policy(g, P, W, POLICIES["TTF-pilot"])` takes a 1-s grid `g` and a dictionary `P` of PERCLOS series (percent) for the 30-s window and the reference window `W`, and returns the FULL time and the list of PRE episodes. The FULL threshold, the PRE threshold of the fixed policy, `p_min` and `tau` are arguments. `common.grid` shows how irregularly sampled PERCLOS is placed on the causal grid.

## Data

### Study

Fifty licensed adult drivers drove a monotonous straight highway section of the Springfield road network in a miniSim driving simulator (Driving Safety Research Institute, University of Iowa) between 10 and 17 April 2026. The drive was stopped when the operator observed drowsiness and did not exceed 30 min. The face was recorded with an RGB camera mounted above the steering wheel (NexiGo N60), and an eye tracker (FOVIO FX3) was recorded at the same time. Eye opening was computed from FaceMesh-V2 landmarks with the FaceAnalyzer package. Four participants have no RGB camera data and are not analysed. The study was approved by the Purdue University Institutional Review Board (IRB-2025-646).

### What is shared

Only data generated or processed by the authors are shared here. In accordance with the Institutional Review Board requirements, study participant numbers were replaced by random codes (S01–S46) that cannot be linked to the original records, and individual demographic information is not shared. Absolute timestamps were replaced by the elapsed time from the first RGB camera sample of each record. Face video, eye-tracker recordings and the uncorrected outputs of the RGB camera pipeline are not included. The eye-tracker recordings and the uncorrected camera outputs are available from the corresponding author on reasonable request, after review and within the scope approved by the Institutional Review Board. Face video is not shared because it identifies the participants.

### `participants_summary.csv`

Aggregated characteristics of the analysed participants: numbers enrolled and analysed, numbers of men and women, mean and SD of age and of driving experience, numbers of morning and afternoon sessions, and counts by self-reported ethnicity.

### `eye/S<code>.csv.gz`

One row per RGB camera frame (about 20.8 Hz). Empty cells are missing values; they are never filled with zeros.

| Column | Description |
|---|---|
| `t` | Elapsed time from the first sample (s) |
| `left_eye_opening`, `right_eye_opening` | Normalized eye opening from MediaPipe face-mesh landmarks after correction (see below) |
| `left_eye_data_source`, `right_eye_data_source` | `webcam` (original measurement), `eyetracker_pupil_open_proxy` (reconstructed as open from eye-tracker pupil tracking), `manual_active` (designated open after video review), `unknown` |
| `eye_state` | `OPEN`, `PARTIAL`, `CLOSED_80` (mean opening < 0.20), `OPEN_ONE_EYE`, `OPEN_MANUAL_ACTIVE`, `UNKNOWN` |
| `eye_metrics_valid` | Whether eye metrics could be computed for the frame |
| `blink_detected` | True on the frame where a blink starts (mean opening < 0.35; reopening at ≥ 0.3675) |
| `blink_duration` | Blink duration (s), written once on the reopening frame |
| `perclos_30s`, `perclos_1min`, `perclos_2min`, `perclos_3min`, `perclos_5min` | Time-weighted PERCLOS (%) over the past 30, 60, 120, 180 and 300 s: closed time divided by valid observation time. Empty when less than 80% of the window is valid or the current state is unknown. Before a full window has elapsed, only the observed time is used; the analysis uses values only from t ≥ W |
| `valid_fraction_30s` … `valid_fraction_5min` | Fraction of the window covered by valid observations |

**Corrections.** RGB camera and eye-tracker samples were matched in time (±30 ms). A missing RGB camera value, or an RGB camera value below 0.35 that lasted at least 0.3 s, was reconstructed as open when the eye tracker tracked the pupil of the same eye normally at that time (continuous tracking of at least 0.1 s for missing values); the reconstructed value is the participant- and eye-specific median of valid open values. These reconstructions cover 13,772 of 2,019,476 eye-side values (0.7%). Segments from two participants (10,876 frames in total, mostly within the first 4–5 min) were designated open because two researchers reviewing the RGB camera video, cross-checked with the eye-tracker data, judged the participants to be active; using the original eye states in these segments instead leaves every warning outcome unchanged. 313 frames could not be classified and are left missing. Because the corrected series uses eye-tracker information, it should not be used to compare the two devices independently.

`preprocessing_checks.py` confirms that the 60-s PERCLOS column can be reproduced from the frame-level eye states (mean absolute difference below 0.0001 percentage points).

### `blink_events.csv`

`participant` (random code), `start_index` and `end_index` (row indices in the participant's eye file), `start_t` and `end_t` (elapsed s), `duration_seconds`, and `long_closure_review` (True for long closures flagged for video review). Blink measures are not analysed in the article.

### `preprocessing_summary.csv`

Per-participant counts (random codes): RGB camera frames, proxy-reconstructed values per eye, manually designated frames, unknown frames and completed blinks after correction.

## Licence

Code: MIT licence (`LICENSE`). Data: CC BY 4.0 (`data/LICENSE.md`).

## Citation

Please cite the article above. Citation metadata are in `CITATION.cff`.

## Contact

Cheonin Oh, Electronics and Telecommunications Research Institute (ETRI), cioh@etri.re.kr
