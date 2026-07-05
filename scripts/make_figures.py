"""Generate the three headline figures for the blog post.

Data is curated from EXPERIMENTS.md (E4/E5/E6/E8/E9). Where we have multiple seeds the
figure shows mean +/- std and the individual points; single-seed points are marked as such.
Outputs PNGs to figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
LGBM = 0.043            # LightGBM baseline PR-AUC (E0)
BLUE, ORANGE, GRAY = "#2b6cb0", "#dd6b20", "#718096"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def baseline(ax):
    ax.axhline(LGBM, ls="--", c=GRAY, lw=1.3)
    ax.text(0.99, LGBM, " LightGBM 0.043", color=GRAY, va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=9)


# ---- Fig 1: scaling with model size (bucket+dt, as-of-date) ----
def fig_scaling():
    params = np.array([2.9, 7.5, 13.7])           # millions
    nano_seeds = [0.2265, 0.1150, 0.1511]          # E9 (3 seeds)
    pr_3k = [np.mean(nano_seeds), 0.364, 0.349]    # nano(mean), mini, small @3000 steps
    nano_std = np.std(nano_seeds, ddof=1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(params, pr_3k, "-o", color=BLUE, lw=2, ms=8, label="3000 steps")
    ax.errorbar(params[0], pr_3k[0], yerr=nano_std, fmt="none", ecolor=BLUE, capsize=4)
    ax.scatter([2.9]*3, nano_seeds, color=BLUE, alpha=0.35, s=25, zorder=3)
    # small @6000 steps (E6)
    ax.scatter([13.7], [0.495], color=ORANGE, s=140, marker="*", zorder=5,
               label="small @6000 steps")
    ax.annotate("undertrained\nat 3k steps", (13.7, 0.349), (9.5, 0.28),
                fontsize=8.5, color=GRAY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate("+3k steps", (13.7, 0.495), (10.2, 0.47), fontsize=8.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    baseline(ax)
    ax.set_xlabel("model size (M parameters)"); ax.set_ylabel("test PR-AUC")
    ax.set_title("(1) Scaling with model size  (fraud, as-of-date)")
    ax.set_ylim(0, 0.55); ax.legend(loc="center right", fontsize=9)
    ax.text(0.02, 0.02, "nano: mean±std over 3 seeds; mini/small single seed",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig1_model_scaling.png"); plt.close(fig)


# ---- Fig 2: scaling with sequence length (nano bucket+dt, as-of-date) ----
def fig_seqlen():
    L = np.array([4, 8, 16, 32, 64, 128, 256])
    pr = np.array([0.055, 0.151, 0.167, 0.212, 0.155, 0.227, 0.235])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(L, pr, "-o", color=BLUE, lw=2, ms=7)
    # flag the noisy L=64 point
    ax.scatter([64], [0.155], facecolors="white", edgecolors=BLUE, s=90, zorder=5)
    ax.annotate("seed noise", (64, 0.155), (64, 0.09), fontsize=8.5, color=GRAY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY))
    baseline(ax)
    ax.set_xscale("log", base=2); ax.set_xticks(L); ax.set_xticklabels(L)
    ax.set_xlabel("context length  L  (events, log scale)"); ax.set_ylabel("test PR-AUC")
    ax.set_title("(2) Scaling with sequence length  (nano)")
    ax.set_ylim(0, 0.30)
    ax.text(0.02, 0.02, "single seed (±0.03–0.05); rises through L=256, not yet plateaued",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig2_seq_length.png"); plt.close(fig)


# ---- Fig 3: effect of the delta-t embedding (nano bucket, as-of-date) ----
def fig_deltat():
    dt_seeds = [0.2265, 0.1150, 0.1511]            # +dt, 3 seeds (E9)
    dt_mean, dt_std = np.mean(dt_seeds), np.std(dt_seeds, ddof=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    x = [0, 1]
    ax.bar(0, 0.219, width=0.55, color=GRAY, alpha=0.7, label="single seed")
    ax.bar(1, dt_mean, width=0.55, color=BLUE, alpha=0.8, yerr=dt_std, capsize=6,
           label="mean ± std (3 seeds)")
    ax.scatter([1, 1, 1], dt_seeds, color="k", alpha=0.6, s=28, zorder=5)
    ax.scatter([0], [0.219], color="k", alpha=0.6, s=28, zorder=5)
    ax.annotate("E4 seed-0\n(lucky draw)", (1, 0.2265), (1.25, 0.30), fontsize=8,
                color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY))
    baseline(ax)
    ax.set_xticks(x); ax.set_xticklabels(["no Δt", "+ Δt field"])
    ax.set_ylabel("test PR-AUC"); ax.set_ylim(0, 0.40)
    ax.set_title("(3) Effect of the Δt (time-since-last) embedding")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.02, "Δt gain is within nano seed noise (±0.05) — suggestive, not established",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig3_delta_t.png"); plt.close(fig)


if __name__ == "__main__":
    fig_scaling(); fig_seqlen(); fig_deltat()
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p}  ({p.stat().st_size//1024} KB)")
