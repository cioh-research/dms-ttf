"""Reproduce all results, tables and figures: python code/run_all.py"""
import time

import figures
import level_sensitivity
import metrics
import parameter_sensitivity
import preprocessing_checks
import replay
import stats_analysis

STEPS = [("Replay of warning policies", replay.main),
         ("Event-level metrics (Table 3, Table 4, Supplementary Table S2)", metrics.main),
         ("Participants, paired tests, mixed-effects models, subgroups", stats_analysis.main),
         ("FULL-level sensitivity (Supplementary Table S4)", level_sensitivity.main),
         ("Parameter sensitivity (Supplementary Table S5)", parameter_sensitivity.main),
         ("Preprocessing checks", preprocessing_checks.main),
         ("Figures 2-4", figures.main)]

if __name__ == "__main__":
    t0 = time.time()
    for name, fn in STEPS:
        print(f"\n=== {name}")
        fn()
    print(f"\nDone in {time.time() - t0:.0f} s. Outputs are in results/ and figures/.")
