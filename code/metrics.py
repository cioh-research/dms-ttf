"""Event-level outcome measures (Eq. 12, Tables 3-4, Supplementary Table S2).

Output: results/summary_metrics.csv
"""
import json
import os

import numpy as np
import pandas as pd

from common import RESULTS, TAU, WINDOWS

_EV = None
_FLAT = None


def events():
    global _EV
    if _EV is None:
        _EV = pd.read_csv(os.path.join(RESULTS, "replay_events.csv"))
    return _EV


def flat():
    """One row per PRE episode."""
    global _FLAT
    if _FLAT is None:
        rec = []
        for _, r in events().iterrows():
            for e in json.loads(r.events):
                rec.append(dict(p=r.p, W=r.W, policy=r.policy, init=r.init, status=e["status"], pre=e["pre"],
                                full=e.get("full"), lead=(e["full"] - e["pre"]) if e["status"] == "matched" else np.nan))
        _FLAT = pd.DataFrame(rec)
    return _FLAT


def summarize(policy, W, init="complete", tau=TAU):
    ev, E = events(), flat()
    s = ev[(ev.policy == policy) & (ev.W == W) & (ev.init == init)]
    assert (s.full.fillna(-1) == s.ref_full.fillna(-1)).all(), "FULL time differs from the reference path"
    elig = s[s.ref_full.notna() & ~s.boundary]
    e = E[(E.policy == policy) & (E.W == W) & (E.init == init)]
    m = e[(e.status == "matched") & e.p.isin(elig.p)]
    lead = m.lead.values
    dev = np.abs(lead - tau)
    return dict(W=W, policy=policy, init=init, n_full=int(s.ref_full.notna().sum()), boundary=int(s.boundary.sum()),
                eligible=len(elig), matched=len(m), coverage=100 * len(m) / max(1, len(elig)),
                median=np.median(lead) if len(lead) else np.nan,
                iqr=(np.percentile(lead, 75) - np.percentile(lead, 25)) if len(lead) else np.nan,
                sd=np.std(lead, ddof=1) if len(lead) > 1 else np.nan,
                mae=dev.mean() if len(dev) else np.nan,
                rmse=np.sqrt(np.mean((lead - tau) ** 2)) if len(lead) else np.nan,
                cancelled=int((e.status == "cancelled").sum()), censored=int((e.status == "censored").sum()),
                unmatched=int((e.status != "matched").sum()))


def leads(policy, W, init="complete"):
    """Lead time of every matched, eligible FULL event (indexed by participant)."""
    ev, E = events(), flat()
    s = ev[(ev.policy == policy) & (ev.W == W) & (ev.init == init)]
    elig = s[s.ref_full.notna() & ~s.boundary].p
    e = E[(E.policy == policy) & (E.W == W) & (E.init == init) & (E.status == "matched") & E.p.isin(elig)]
    return e.set_index("p").lead


def main():
    from replay import POLICIES
    rows = []
    for W in WINDOWS:
        for pol in POLICIES:
            rows.append(summarize(pol, W))
        rows.append(summarize("TTF-pilot", W, "startup"))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(RESULTS, "summary_metrics.csv"), index=False)
    # Supplementary Table S1: lead time of every reference FULL event
    ev = events()
    ref = ev[(ev.policy == "NHTSA-replay") & ev.ref_full.notna()]
    rec = []
    for _, r in ref.iterrows():
        row = dict(participant=r.p, W=r.W, full_time_s=r.ref_full, excluded_first_window=bool(r.boundary))
        for pol in ("NHTSA-replay", "TTF-pilot"):
            lt = leads(pol, r.W)
            row[f"lead_{pol}_s"] = lt.get(r.p, np.nan)
        rec.append(row)
    pd.DataFrame(rec).sort_values(["W", "participant"], key=lambda c: c.map(lambda v: int(v[1:]) if isinstance(v, str) else v)) \
        .to_csv(os.path.join(RESULTS, "event_leads.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S[["W", "policy", "init", "eligible", "matched", "median", "iqr", "sd", "mae", "rmse", "unmatched"]].round(1).to_string())


if __name__ == "__main__":
    main()
