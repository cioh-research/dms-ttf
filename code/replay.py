"""Replay of staged PERCLOS warning policies on the 1-s causal grid (Methods, Table 2).

FULL is always evaluated first on the unsmoothed P_W (reference path, Eq. 10), so every
policy issues FULL at the same time. Output: results/replay_events.csv
"""
import json
import math
import os

import numpy as np
import pandas as pd

from common import FULL, LAM, NHTSA_PRE, PMIN, RESULTS, TAU, WINDOWS, cmed, load_grid, participants_with_eye_data

POLICIES = {
    "NHTSA-replay":         dict(kind="level", h=1, r=5, cool=0),
    "NHTSA-replay-matched": dict(kind="level", h=1, r=20, cool=15),
    "TTF-pilot":            dict(kind="ttf", h=1, r=20, cool=15, order=1, rate="max"),
    "TTF-pilot-slow-only":  dict(kind="ttf", h=1, r=20, cool=15, order=1, rate="slow"),
    "TTF-pilot-fast-only":  dict(kind="ttf", h=1, r=20, cool=15, order=1, rate="fast"),
    "TTF-pilot-2nd-order":  dict(kind="ttf", h=1, r=20, cool=15, order=2, rate="max"),
    "TTF-pilot-3rd-order":  dict(kind="ttf", h=1, r=20, cool=15, order=3, rate="max"),
}


def residual(order, p_now, f, v, a=0.0, j=0.0):
    """Smallest positive root R of v R + a R^2/2 + j R^3/6 = f - p_now (Eqs. 8-9); +inf if none."""
    d = f - p_now
    if order == 1 or (a == 0 and j == 0):
        return d / v if v > 0 else math.inf
    coeffs = [j / 6.0, a / 2.0, v, -d] if order == 3 else [a / 2.0, v, -d]
    roots = np.roots(coeffs)
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    return float(roots.min()) if len(roots) else math.inf


def run_policy(g, P, W, policy, init="complete", f=FULL, a_thr=None, pmin=PMIN, tau=TAU):
    """Replay one policy on one record and window. Returns (FULL time or None, list of PRE episodes)."""
    a_thr = NHTSA_PRE[W] if a_thr is None else a_thr
    PW, P30 = P[W], P[30]
    tPW, tP30 = cmed(PW), cmed(P30)
    kind, h, r, cool = policy["kind"], policy["h"], policy["r"], policy["cool"]
    order, rate = policy.get("order", 1), policy.get("rate", "max")
    history = {1: 24, 2: 44, 3: 64}[order] if init == "complete" else 0
    start_pred = W + history
    state, on, off, last_cancel = "NORMAL", 0, 0, -1e9
    events, cur, full_t = [], None, None
    for k, t in enumerate(g):
        if t < W:
            continue
        # reference path: FULL on the unsmoothed signal, evaluated before any prediction
        if not np.isnan(PW[k]) and PW[k] >= f:
            full_t = t
            if state == "PRE":
                cur["status"], cur["full"] = "matched", t
                events.append(cur)
                cur = None
            break
        cand = False
        if kind == "level":
            if not np.isnan(PW[k]):
                cand = PW[k] >= a_thr
        elif t >= start_pred:
            def at(dt):
                i = k - dt
                return tPW[i] if i >= 0 else np.nan
            p0, p20, q30 = at(0), at(20), tP30[k]
            if not (np.isnan(p0) or np.isnan(p20) or np.isnan(q30)):
                vs = (p0 - p20) / 20.0                                  # Eq. (5), slow term
                vf = LAM * max(0.0, q30 - p0) / (W - 30)                # Eq. (5), fast term
                v = {"max": max(0.0, vs, vf), "slow": max(0.0, vs), "fast": max(0.0, vf)}[rate]  # Eq. (6)
                acc = jerk = 0.0
                ok = True
                if order >= 2:
                    p40 = at(40)
                    if np.isnan(p40):
                        ok = False
                    else:
                        acc = ((p0 - p20) - (p20 - p40)) / 400.0
                if order >= 3 and ok:
                    p60 = at(60)
                    if np.isnan(p60):
                        ok = False
                    else:
                        jerk = (p0 - 3 * p20 + 3 * p40 - p60) / 8000.0
                if ok and 0 <= p0 < f:
                    R = residual(order, p0, f, v, acc, jerk)
                    cand = (p0 >= pmin) and (0 <= R <= tau)
        if cand:
            on, off = on + 1, 0
        else:
            off, on = off + 1, 0
        if state == "PRE" and off >= r:
            cur["status"], cur["end"] = "cancelled", t
            events.append(cur)
            cur, state, last_cancel = None, "NORMAL", t
        elif state == "NORMAL" and on >= h and (t - last_cancel) >= cool:
            state, cur = "PRE", {"pre": t}
    if state == "PRE" and cur is not None:
        cur["status"] = "censored"
        events.append(cur)
    return full_t, events


def reference_full(g, P, W, f=FULL):
    for k, t in enumerate(g):
        if t >= W and not np.isnan(P[W][k]) and P[W][k] >= f:
            return t
    return None


def main():
    rows = []
    for p in participants_with_eye_data():
        g, P = load_grid(p)
        for W in WINDOWS:
            ref = reference_full(g, P, W)
            for name, pol in POLICIES.items():
                inits = ["complete", "startup"] if name == "TTF-pilot" else ["complete"]
                for init in inits:
                    full_t, ev = run_policy(g, P, W, pol, init)
                    rows.append(dict(p=p, W=W, policy=name, init=init, ref_full=ref, full=full_t,
                                     boundary=(ref is not None and ref <= W), dur=g[-1], events=json.dumps(ev)))
        print(p, end=" ", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "replay_events.csv"), index=False)
    print(f"\nreplay_events.csv: {len(out)} rows")


if __name__ == "__main__":
    main()
