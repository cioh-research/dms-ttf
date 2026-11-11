"""Post hoc sensitivity to the FULL reference level (Supplementary Table S4).

FULL threshold f in {8, 10, 12}%; the PRE threshold a_W and the TTF lower level p_min are scaled by f/12.
Outputs: results/levels_events.csv, results/levels_table.csv
"""
import os

import numpy as np
import pandas as pd

from common import NHTSA_PRE, PMIN, RESULTS, TAU, WINDOWS, load_grid, participants_with_eye_data
from replay import POLICIES, run_policy

LEVELS = (8.0, 10.0, 12.0)


def main():
    rows = []
    for p in participants_with_eye_data():
        g, P = load_grid(p)
        for W in WINDOWS:
            for f in LEVELS:
                for name in ("NHTSA-replay", "TTF-pilot"):
                    full, ev = run_policy(g, P, W, POLICIES[name], "complete", f=f,
                                          a_thr=NHTSA_PRE[W] * f / 12.0, pmin=PMIN * f / 12.0)
                    m = [e for e in ev if e["status"] == "matched"]
                    rows.append(dict(p=p, W=W, f=f, policy=name, full=full, boundary=(full is not None and full <= W),
                                     lead=(full - m[0]["pre"]) if m else np.nan,
                                     unmatched=sum(e["status"] != "matched" for e in ev)))
        print(p, end=" ", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RESULTS, "levels_events.csv"), index=False)
    tab = []
    for (W, f), d in D.groupby(["W", "f"]):
        a = d[d.policy == "NHTSA-replay"].set_index("p")
        b = d[d.policy == "TTF-pilot"].set_index("p")
        el = a[a.full.notna() & ~a.boundary].index
        la, lb = a.loc[el, "lead"].dropna(), b.loc[el, "lead"].dropna()
        tab.append(dict(W=W, f=f, full=len(el), m_a=len(la), sd_a=la.std(), e_a=(la - TAU).abs().mean(), u_a=int(a.unmatched.sum()),
                        m_b=len(lb), sd_b=lb.std(), e_b=(lb - TAU).abs().mean(), u_b=int(b.unmatched.sum())))
    T = pd.DataFrame(tab)
    T.to_csv(os.path.join(RESULTS, "levels_table.csv"), index=False)
    print("\n", T.round(1).to_string())


if __name__ == "__main__":
    main()
