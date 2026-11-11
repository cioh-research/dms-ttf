"""Shared paths, constants and data loading."""
import glob
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EYE = os.path.join(DATA, "eye")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# PERCLOS columns (percent) for each window length in seconds
COL = {30: "perclos_30s", 60: "perclos_1min", 180: "perclos_3min", 300: "perclos_5min"}
WINDOWS = (60, 180, 300)
NHTSA_PRE = {60: 8.0, 180: 9.0, 300: 10.0}   # NHTSA PRE thresholds a_W (%)
FULL = 12.0                                    # NHTSA FULL threshold f_W (%)
TAU = 45.0                                     # target lead time (s)
PMIN = 5.0                                     # minimum PRE-eligibility level (%)
LAM = 2.0                                      # short-window fusion coefficient, Eq. (7)
HOLD = 2                                       # maximum zero-order hold across missing samples (s)


def participants_with_eye_data():
    files = glob.glob(os.path.join(EYE, "S*.csv.gz"))
    return sorted(os.path.basename(f).split(".")[0] for f in files)


def load(p, cols=None):
    """Eye-state time series of participant p; column t is elapsed time (s) from the first sample."""
    use = None if cols is None else ["t"] + [c for c in cols if c != "t"]
    return pd.read_csv(os.path.join(EYE, f"{p}.csv.gz"), usecols=use, low_memory=False)


RESTRICTED = os.path.join(DATA, "restricted", "participants.csv")


def demographics():
    """Individual participant characteristics. Not distributed (IRB restriction); returns None if absent."""
    return pd.read_csv(RESTRICTED) if os.path.exists(RESTRICTED) else None


def grid(df):
    """1-s causal grid: value at k is the last valid sample with t <= k, held for at most HOLD s."""
    T = int(math.floor(df["t"].iloc[-1]))
    g = np.arange(0, T + 1, dtype=float)
    out = {}
    for w, c in COL.items():
        s = pd.to_numeric(df[c], errors="coerce")
        ok = s.notna().values
        tt = df["t"].values[ok]
        vv = s.values[ok]
        idx = np.searchsorted(tt, g, side="right") - 1
        val = np.where(idx >= 0, vv[np.clip(idx, 0, None)], np.nan)
        age = np.where(idx >= 0, g - tt[np.clip(idx, 0, None)], np.inf)
        val[age > HOLD] = np.nan
        out[w] = val
    return g, out


def load_grid(p):
    return grid(load(p, list(COL.values())))


def cmed(x, k=5):
    """Causal median of the k most recent samples (NaN if any is missing)."""
    return pd.Series(x).rolling(k, min_periods=k).median().values
