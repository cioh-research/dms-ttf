"""Checks of the eye-state preprocessing reported in Methods.

Frame rate, blink-onset threshold, closure threshold, recomputation of 60-s PERCLOS from sample states,
correction counts and the minimum threshold-passage interval (Eq. 4).
Output: results/preprocessing_checks.json
"""
import json
import os

import numpy as np
import pandas as pd

from common import DATA, FULL, NHTSA_PRE, RESULTS, WINDOWS, load, participants_with_eye_data

BLINK_START, CLOSED = 0.35, 0.20


def perclos_from_states(df, W, min_valid=0.8):
    """Time-weighted PERCLOS over (t-W, t] at each sample; a sample's state holds over (t_{i-1}, t_i]."""
    t = df.t.values
    dt = np.diff(t, prepend=t[0])
    ok_gap = (dt > 0) & (dt <= max(0.10, 3 * np.median(np.diff(t))))
    valid = (df.eye_state.astype(str) != "UNKNOWN").values & (df.eye_metrics_valid.astype(str).str.lower() == "true").values
    closed = (df.eye_state.astype(str) == "CLOSED_80").values
    cv = np.cumsum(np.where(ok_gap & valid, dt, 0.0))
    cc = np.cumsum(np.where(ok_gap & valid & closed, dt, 0.0))
    v0 = np.interp(t - W, t, cv, left=0.0)
    c0 = np.interp(t - W, t, cc, left=0.0)
    vw, cw = cv - v0, cc - c0
    out = np.where(vw > 0, 100 * cw / np.maximum(vw, 1e-9), np.nan)
    out[(vw < min_valid * W) | (t < W)] = np.nan
    return out


def main():
    res = {}
    fps, onsets, onset_ok, closed_rows, closed_ok, errs = [], 0, 0, 0, 0, []
    for p in participants_with_eye_data():
        df = load(p)
        fps.append(1 / np.median(np.diff(df.t.values)))
        L = pd.to_numeric(df.left_eye_opening, errors="coerce")
        R = pd.to_numeric(df.right_eye_opening, errors="coerce")
        both = (L >= 0) & (R >= 0)
        mean_open = (L + R) / 2
        on = df.blink_detected.astype(str).str.lower() == "true"
        onsets += int(on.sum())
        onset_ok += int((on & (mean_open < BLINK_START)).sum())
        cl = df.eye_state.astype(str) == "CLOSED_80"
        closed_rows += int(cl.sum())
        closed_ok += int((cl & both & (mean_open < CLOSED)).sum())
        pw = perclos_from_states(df, 60)
        ref = pd.to_numeric(df.perclos_1min, errors="coerce").values
        m = ~np.isnan(pw) & ~np.isnan(ref)
        errs.append(np.abs(pw[m] - ref[m]))
    e = np.concatenate(errs)
    res["frame_rate_hz"] = dict(mean=float(np.mean(fps)), min=float(np.min(fps)), max=float(np.max(fps)))
    res["blink_onsets"] = dict(n=onsets, mean_opening_below_0_35=onset_ok)
    res["closed_80_rows"] = dict(n=closed_rows, mean_opening_below_0_20=closed_ok)
    res["perclos_60s_recomputed_abs_error_pp"] = dict(mean=float(e.mean()), p99=float(np.percentile(e, 99)))
    cs = pd.read_csv(os.path.join(DATA, "preprocessing_summary.csv"))
    rows = int(cs.webcam_rows.sum())
    proxy = int(cs.left_proxy_reconstructed_rows.sum() + cs.right_proxy_reconstructed_rows.sum())
    res["corrections"] = dict(samples=rows, eye_side_values=2 * rows, proxy_reconstructed_eye_side_values=proxy,
                              proxy_pct_of_eye_side_values=100 * proxy / (2 * rows),
                              manual_active_samples=int(cs.manual_active_rows.sum()), unknown_samples=int(cs.unknown_eye_rows.sum()))
    res["delta_min_s"] = {str(W): W * (FULL - NHTSA_PRE[W]) / 100 for W in WINDOWS}
    json.dump(res, open(os.path.join(RESULTS, "preprocessing_checks.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
