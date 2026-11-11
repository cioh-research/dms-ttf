"""One-at-a-time parameter sensitivity of TTF-pilot against NHTSA-replay (Supplementary Table S5).

Output: results/param_sens.csv
"""
import math
from decimal import Decimal, ROUND_HALF_UP
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import FULL, LAM, NHTSA_PRE, RESULTS, WINDOWS, load_grid, participants_with_eye_data

warnings.filterwarnings("ignore")


def f1(x):
    """One decimal, rounding halves up (as in the article tables)."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def cmed(x, k):
    return pd.Series(x).rolling(k, min_periods=k).median().values


def run(g, P, W, kind, tau=45, pmin=5.0, h=1, r=20, cool=15, look=20, med=5):
    """First-order replay with configurable parameters; identical to replay.run_policy at the defaults."""
    PW, P30 = P[W], P[30]
    tPW, tP30 = cmed(PW, med), cmed(P30, med)
    start = W + look + med - 1
    state, on, off, last, cur, ev, full = "NORMAL", 0, 0, -1e9, None, [], None
    for k, t in enumerate(g):
        if t < W:
            continue
        if not np.isnan(PW[k]) and PW[k] >= FULL:
            full = t
            if state == "PRE":
                cur.update(status="matched", full=t)
                ev.append(cur)
            break
        c = False
        if kind == "level":
            c = (not np.isnan(PW[k])) and PW[k] >= NHTSA_PRE[W]
        elif t >= start:
            p0, pb, q = tPW[k], tPW[k - look], tP30[k]
            if not (np.isnan(p0) or np.isnan(pb) or np.isnan(q)) and 0 <= p0 < FULL:
                v = max(0.0, (p0 - pb) / look, LAM * max(0.0, q - p0) / (W - 30))
                R = (FULL - p0) / v if v > 0 else math.inf
                c = (p0 >= pmin) and (R <= tau)
        if c:
            on, off = on + 1, 0
        else:
            off, on = off + 1, 0
        if state == "PRE" and off >= r:
            cur.update(status="cancelled")
            ev.append(cur)
            cur, state, last = None, "NORMAL", t
        elif state == "NORMAL" and on >= h and t - last >= cool:
            state, cur = "PRE", {"pre": t}
    if state == "PRE" and cur and full is None:
        cur["status"] = "censored"
        ev.append(cur)
    m = [e for e in ev if e["status"] == "matched"]
    return full, (full - m[0]["pre"]) if m else np.nan, sum(e["status"] != "matched" for e in ev)


DEFAULT = dict(tau=45, pmin=5.0, h=1, r=20, cool=15, look=20, med=5)
VARY = {"tau": [30, 60], "pmin": [3.0, 7.0], "h": [3, 5], "r": [10, 30], "cool": [0, 30], "look": [10, 30], "med": [3, 7]}


def main():
    configs = [("default", DEFAULT)] + [(f"{k}={v}", {**DEFAULT, k: v}) for k, vs in VARY.items() for v in vs]
    sig = {p: load_grid(p) for p in participants_with_eye_data()}
    base = {(p, W): run(g, P, W, "level", r=5, h=1, cool=0) for p, (g, P) in sig.items() for W in WINDOWS}
    rows = []
    for name, cfg in configs:
        tau = cfg["tau"]
        rec = []
        for p, (g, P) in sig.items():
            for W in WINDOWS:
                fb, lb, _ = base[(p, W)]
                if fb is None or fb <= W:
                    continue
                ft, lt, _ = run(g, P, W, "ttf", **cfg)
                assert ft == fb
                rec.append(dict(p=p, W=W, lthr=lb, lttf=lt))
        R = pd.DataFrame(rec)
        out = dict(config=name)
        sd_better = e_better = 0
        for W in WINDOWS:
            d = R[R.W == W]
            a, b = d.lthr.dropna(), d.lttf.dropna()
            out[f"cov{W}"] = f"{len(b)}/{len(d)}"
            out[f"sd{W}"] = f"{f1(a.std())}->{f1(b.std())}"
            out[f"e{W}"] = f"{f1((a - tau).abs().mean())}->{f1((b - tau).abs().mean())}"
            sd_better += b.std() < a.std()
            e_better += (b - tau).abs().mean() < (a - tau).abs().mean()
        L = pd.concat([pd.DataFrame(dict(p=R.p, W=R.W.astype(str), pol="Thr", dev=(R.lthr - tau).abs())),
                       pd.DataFrame(dict(p=R.p, W=R.W.astype(str), pol="TTF", dev=(R.lttf - tau).abs()))]).dropna()
        m = smf.mixedlm("dev ~ C(pol, Treatment('Thr')) + C(W)", L, groups=L["p"]).fit(reml=True)
        k = [c for c in m.params.index if "pol" in c][0]
        ci = m.conf_int().loc[k].values
        out.update(sd_better=f"{sd_better}/3", e_better=f"{e_better}/3", effect=f1(m.params[k]),
                   ci=f"[{f1(ci[0])}, {f1(ci[1])}]", P=round(m.pvalues[k], 4))
        rows.append(out)
        print(out, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "param_sens.csv"), index=False)


if __name__ == "__main__":
    main()
