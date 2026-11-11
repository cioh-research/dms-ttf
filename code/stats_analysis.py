"""Paired tests (Table 3), mixed-effects models, and, when the restricted participant file is present,
participant characteristics (Table 1) and group comparisons (Fig. 4).

Individual demographics are not distributed (IRB restriction). Without data/restricted/participants.csv,
Table 1 and the group comparisons are skipped and the aggregated results already in results/ are kept.

Outputs: results/paired_<policy>.csv, results/mixed_models.json, results/paired_events_ttf.csv,
         results/table1_participants.json and results/subgroups.csv (aggregated; restricted file required)
"""
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import RESULTS, TAU, WINDOWS, demographics, load, participants_with_eye_data
from metrics import events, leads

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)


def describe(d):
    """Aggregated characteristics only (no individual values or ranges)."""
    return dict(n=len(d), age_mean=d.age.mean(), age_sd=d.age.std(),
                exp_mean=d.driving_experience_years.mean(), exp_sd=d.driving_experience_years.std(),
                male=int((d.gender == "Male").sum()), female=int((d.gender == "Female").sum()),
                morning=int((d.session == "Morning").sum()), afternoon=int((d.session == "Afternoon").sum()),
                dur_mean=d.dur_min.mean(), dur_sd=d.dur_min.std(), total_h=d.dur_min.sum() / 60)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0
    for i, k in enumerate(order):
        running = max(running, (m - i) * ps[k])
        adj[k] = min(1, running)
    return adj


def boot_ci(f, a, b, B=20000):
    """Participant-level paired percentile bootstrap."""
    n = len(a)
    vals = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        vals[i] = f(a[idx], b[idx])
    return np.percentile(vals, [2.5, 97.5])


def table1(ev):
    dem = demographics().copy()
    dur = {p: load(p, ["t"]).t.iloc[-1] / 60 for p in participants_with_eye_data()}
    dem["dur_min"] = dem.participant.map(dur)
    dem = dem.rename(columns={"participant": "p"})
    out = {"all": describe(dem)}
    out["all"]["ethnicity"] = dem.ethnicity.value_counts().to_dict()
    ref = ev[ev.policy == "NHTSA-replay"]
    for W in WINDOWS:
        s = ref[ref.W == W]
        full_p = s[s.ref_full.notna() & ~s.boundary].p
        d = dem[dem.p.isin(full_p)]
        nd = dem[~dem.p.isin(s[s.ref_full.notna()].p)]
        out[f"full_{W}"] = describe(d)
        out[f"not_full_{W}"] = describe(nd)
        out[f"tests_{W}"] = dict(
            age_p=stats.mannwhitneyu(d.age, nd.age).pvalue,
            exp_p=stats.mannwhitneyu(d.driving_experience_years, nd.driving_experience_years).pvalue,
            dur_p=stats.mannwhitneyu(d.dur_min, nd.dur_min).pvalue,
            gender_p=stats.fisher_exact(pd.crosstab(dem.p.isin(full_p), dem.gender == "Male").values)[1],
            session_p=stats.fisher_exact(pd.crosstab(dem.p.isin(full_p), dem.session == "Morning").values)[1])
    return dem, out


def paired_tests():
    res_all = {}
    for pol in ["TTF-pilot", "TTF-pilot-2nd-order", "TTF-pilot-3rd-order", "TTF-pilot-slow-only",
                "TTF-pilot-fast-only", "NHTSA-replay-matched"]:
        res = []
        for W in WINDOWS:
            a, b = leads("NHTSA-replay", W), leads(pol, W)
            c = a.index.intersection(b.index)
            A_, B_ = a[c].values, b[c].values
            da, db = np.abs(A_ - TAU), np.abs(B_ - TAU)
            w = stats.wilcoxon(db, da, method="exact")
            ci = boot_ci(lambda x, y: np.mean(np.abs(y - TAU) - np.abs(x - TAU)), A_, B_)
            sdci = boot_ci(lambda x, y: np.log(np.std(y, ddof=1) / np.std(x, ddof=1))
                           if np.std(x) > 0 and np.std(y) > 0 else 0.0, A_, B_)
            res.append(dict(W=W, n=len(c), mae_nhtsa=da.mean(), mae_pol=db.mean(), diff=(db - da).mean(),
                            ci_lo=ci[0], ci_hi=ci[1], p_wilcoxon=w.pvalue, sd_nhtsa=A_.std(ddof=1), sd_pol=B_.std(ddof=1),
                            sd_ratio=B_.std(ddof=1) / A_.std(ddof=1), sd_ratio_lo=math.exp(sdci[0]), sd_ratio_hi=math.exp(sdci[1]),
                            rmse_nhtsa=np.sqrt(np.mean(da ** 2)), rmse_pol=np.sqrt(np.mean(db ** 2))))
        r = pd.DataFrame(res)
        r["p_wilcoxon_holm"] = holm(r.p_wilcoxon.values)
        r.to_csv(os.path.join(RESULTS, f"paired_{pol}.csv"), index=False)
        res_all[pol] = r
    return res_all


def mixed_models():
    rows = []
    for W in WINDOWS:
        for pol in ("NHTSA-replay", "TTF-pilot"):
            for p, v in leads(pol, W).items():
                rows.append(dict(p=p, W=str(W), policy="TTF" if pol == "TTF-pilot" else "Threshold", lead=v, dev=abs(v - TAU)))
    L = pd.DataFrame(rows)
    out = {}
    m = smf.mixedlm("dev ~ C(policy, Treatment('Threshold')) + C(W)", L, groups=L["p"]).fit(reml=True)
    k = [c for c in m.params.index if "policy" in c][0]
    ci = m.conf_int().loc[k].values
    out["absolute_error"] = dict(n_obs=len(L), n_participants=L.p.nunique(), effect_s=m.params[k], ci_lo=ci[0], ci_hi=ci[1], p=m.pvalues[k])
    L["ldev"] = np.log1p(L.dev)
    m = smf.mixedlm("ldev ~ C(policy, Treatment('Threshold')) + C(W)", L, groups=L["p"]).fit(reml=True)
    ci = m.conf_int().loc[k].values
    out["log_error"] = dict(ratio=math.exp(m.params[k]), ci_lo=math.exp(ci[0]), ci_hi=math.exp(ci[1]), p=m.pvalues[k])
    return out


def paired_events():
    rows = []
    for W in WINDOWS:
        a, b = leads("NHTSA-replay", W), leads("TTF-pilot", W)
        for p in a.index.intersection(b.index):
            rows.append((p, W, abs(a[p] - TAU), abs(b[p] - TAU), a[p], b[p]))
    d = pd.DataFrame(rows, columns=["p", "W", "da", "db", "la", "lb"])
    d.to_csv(os.path.join(RESULTS, "paired_events_ttf.csv"), index=False)
    closer, farther = int((d.db < d.da).sum()), int((d.db > d.da).sum())
    pooled = dict(n_participants=d.p.nunique(), n_events=len(d), closer=closer, farther=farther,
                  equal=int((d.db == d.da).sum()), sign_test_p=stats.binomtest(closer, closer + farther).pvalue)
    return d, pooled


def subgroups(d, dem):
    """Group means of participant-level absolute deviation (aggregated output only)."""
    dem = dem.rename(columns={"participant": "p"})
    d = d.merge(dem[["p", "gender", "age", "driving_experience_years", "session"]], on="p")
    d["age_grp"] = np.where(d.age <= 21, "<=21", ">=22")
    d["exp_grp"] = np.where(d.driving_experience_years < 5, "<5 y", ">=5 y")
    gm = d.groupby("p").agg(da=("da", "mean"), db=("db", "mean"), gender=("gender", "first"), age_grp=("age_grp", "first"),
                            exp_grp=("exp_grp", "first"), session=("session", "first"))
    sub = []
    for col in ["gender", "age_grp", "exp_grp", "session"]:
        levels = sorted(gm[col].unique())
        x, y = gm[gm[col] == levels[0]], gm[gm[col] == levels[1]]
        for lv, s_ in ((levels[0], x), (levels[1], y)):
            sub.append(dict(factor=col, level=lv, n_participants=len(s_),
                            mean_dev_nhtsa=s_.da.mean(), sd_dev_nhtsa=s_.da.std(), mean_dev_ttf=s_.db.mean(), sd_dev_ttf=s_.db.std(),
                            p_nhtsa=stats.mannwhitneyu(x.da, y.da).pvalue, p_ttf=stats.mannwhitneyu(x.db, y.db).pvalue))
    sub = pd.DataFrame(sub)
    sub.to_csv(os.path.join(RESULTS, "subgroups.csv"), index=False)
    return sub


def main():
    ev = events()
    paired = paired_tests()
    mm = mixed_models()
    d, pooled = paired_events()
    mm["pooled_events"] = pooled
    json.dump(mm, open(os.path.join(RESULTS, "mixed_models.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print("Paired NHTSA-replay vs TTF-pilot:\n", paired["TTF-pilot"].round(3).to_string())
    print("\nMixed models:", json.dumps(mm, indent=1, default=float))
    dem = demographics()
    if dem is None:
        print("\nRestricted participant file not found: Table 1 and group comparisons skipped "
              "(aggregated results are in results/table1_participants.json and results/subgroups.csv).")
        return
    _, t1 = table1(ev)
    json.dump(t1, open(os.path.join(RESULTS, "table1_participants.json"), "w"), indent=1, default=float)
    sub = subgroups(d, dem)
    print("Tests, W = 60 s:", {k: round(v, 3) for k, v in t1["tests_60"].items()})
    print("\nGroups (participant means):\n", sub.round(3).to_string())


if __name__ == "__main__":
    main()
