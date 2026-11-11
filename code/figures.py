"""Figures 2-4 of the manuscript. Outputs: figures/fig2_leadtime.png, fig3_examples.png, fig4_subgroups.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import FIGURES, FULL, NHTSA_PRE, RESULTS, TAU, load_grid
from metrics import flat, leads

C_N, C_T = "#eb6834", "#2a78d6"     # NHTSA-replay (orange), TTF-pilot (blue)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", color=INK)


def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9), sharey=True)
    for ax, W, lab in zip(axes, (60, 180, 300), "abc"):
        a, b = leads("NHTSA-replay", W), leads("TTF-pilot", W)
        for p in a.index.intersection(b.index):
            ax.plot([0, 1], [a[p], b[p]], color=GRID, lw=0.8, zorder=1)
        for x, s, col, mk in ((0, a, C_N, "o"), (1, b, C_T, "^")):
            q1, q2, q3 = np.percentile(s.values, [25, 50, 75])
            ax.add_patch(plt.Rectangle((x - 0.18, q1), 0.36, q3 - q1, fill=False, ec=col, lw=1.0, zorder=2))
            ax.plot([x - 0.18, x + 0.18], [q2, q2], color=col, lw=2, zorder=3)
            jit = np.linspace(-0.07, 0.07, len(s)) if len(s) > 1 else [0]
            ax.scatter(x + np.array(jit), s.values, s=22, marker=mk, color=col, ec="white", lw=0.6, zorder=4)
            ax.text(x, -14, f"n = {len(s)}", ha="center", va="top", color=INK2, fontsize=7.5)
        ax.axhline(TAU, color=INK2, ls=(0, (4, 3)), lw=0.8, zorder=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["NHTSA-\nreplay", "TTF-\npilot"])
        ax.set_xlim(-0.55, 1.55); ax.set_ylim(-22, 172)
        ax.set_title(f"$W$ = {W} s", fontsize=8.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        panel_label(ax, lab)
    axes[0].set_ylabel("PRE-to-FULL lead time (s)")
    axes[1].text(1.55, TAU + 3, "Target $\\tau$ = 45 s", ha="right", va="bottom", color=INK2, fontsize=7.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(FIGURES, "fig2_leadtime.png"), dpi=300)
    plt.close(fig)


def trace(ax, p, W, lab, title):
    g, P = load_grid(p)
    E = flat()
    e = E[(E.p == p) & (E.W == W) & (E.status == "matched") & (E.init == "complete")]
    tn = e[e.policy == "NHTSA-replay"].pre.iloc[0]
    tt = e[e.policy == "TTF-pilot"].pre.iloc[0]
    tf = e.full.iloc[0]
    lo, hi = max(W, min(tn, tt) - 60), tf + 25
    m = (g >= lo) & (g <= hi)
    ax.plot(g[m], P[30][m], color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="Short-window PERCLOS (30 s)")
    ax.plot(g[m], P[W][m], color=INK, lw=1.4, label="Reference PERCLOS (window $W$)")
    ax.axhline(FULL, color=INK2, lw=0.7, ls=(0, (4, 3)))
    ax.axhline(NHTSA_PRE[W], color=INK2, lw=0.7, ls=(0, (6, 2, 1, 2)))
    bb = dict(fc="white", ec="none", pad=0.4)
    ax.text(lo + 1, FULL + 0.5, "FULL threshold", ha="left", va="bottom", fontsize=7, color=INK2, bbox=bb)
    ax.text(lo + 1, NHTSA_PRE[W] + 0.5, "NHTSA PRE threshold", ha="left", va="bottom", fontsize=7, color=INK2, bbox=bb)
    ymax = 24
    marks = [(tn, C_N, f"NHTSA-replay PRE\n{tn:.0f} s"), (tt, C_T, f"TTF-pilot PRE\n{tt:.0f} s"), (tf, INK, f"FULL\n{tf:.0f} s")]
    for x, col, txt in marks:
        ax.axvline(x, color=col, lw=1.2)
        near = [y for y, _, _ in marks if y != x and abs(y - x) < 0.12 * (hi - lo)]
        ha = "center" if not near else ("right" if x < near[0] else "left")
        dx = {"center": 0, "right": -0.6, "left": 0.6}[ha]
        ax.text(x + dx, ymax - 0.3, txt, ha=ha, va="top", fontsize=7, color=INK, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_ylim(0, ymax); ax.set_xlim(lo, hi)
    ax.set_xlabel("Elapsed time (s)"); ax.set_ylabel("PERCLOS (%)")
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=10)
    panel_label(ax, lab)
    return tn, tt, tf


def fig3():
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.0))
    r1 = trace(axes[0], "S28", 300, "a", "$W$ = 300 s")
    r2 = trace(axes[1], "S29", 60, "b", "$W$ = 60 s")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    fig.savefig(os.path.join(FIGURES, "fig3_examples.png"), dpi=300)
    plt.close(fig)
    print("Fig. 3 (NHTSA PRE, TTF PRE, FULL):", r1, r2)


def fig4():
    """Group means (+/- SD) of participant-level absolute deviation; aggregated values only."""
    g = pd.read_csv(os.path.join(RESULTS, "subgroups.csv"))
    cats = [("gender", "Female"), ("gender", "Male"), ("age_grp", "<=21"), ("age_grp", ">=22"),
            ("exp_grp", "<5 y"), ("exp_grp", ">=5 y"), ("session", "Morning"), ("session", "Afternoon")]
    labels = ["Female", "Male", "≤21 y", "≥22 y", "<5 y", "≥5 y", "Morning", "Afternoon"]
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    for i, (col, lv) in enumerate(cats):
        r = g[(g.factor == col) & (g.level == lv)].iloc[0]
        for off, m, sd, c, mk in ((-0.15, r.mean_dev_nhtsa, r.sd_dev_nhtsa, C_N, "o"), (0.15, r.mean_dev_ttf, r.sd_dev_ttf, C_T, "^")):
            ax.errorbar(i + off, m, yerr=0 if np.isnan(sd) else sd, fmt=mk, color=c, ms=5, capsize=3, lw=1.2, zorder=3)
        ax.text(i, -6, f"n = {int(r.n_participants)}", ha="center", va="top", fontsize=7, color=INK2)
    for x in (1.5, 3.5, 5.5):
        ax.axvline(x, color=GRID, lw=0.6)
    for x, t in ((0.5, "Gender"), (2.5, "Age"), (4.5, "Driving experience"), (6.5, "Session start")):
        ax.text(x, 158, t, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(8)); ax.set_xticklabels(labels)
    ax.set_ylim(-14, 168); ax.set_xlim(-0.6, 7.6)
    ax.set_ylabel("|Lead time − $\\tau$| (s)\ngroup mean ± SD")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_N, ls="none", ms=5, label="NHTSA-replay"),
                       Line2D([], [], marker="^", color=C_T, ls="none", ms=5, label="TTF-pilot")],
              loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig4_subgroups.png"), dpi=300)
    plt.close(fig)


def main():
    fig2()
    fig3()
    fig4()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
