"""Generate figures from the method-(a) sliding-window as-of-date results (eval_*_swasof.json).

All numbers are on the same stratified test subsample (base rate ~1.9%). Missing files are
skipped gracefully, so this can be run before L=256 lands and re-run after.
Outputs PNGs to figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("artifacts"); OUT = Path("figures"); OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY = "#2b6cb0", "#dd6b20", "#718096"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def load(name):
    f = ART / f"eval_{name}_swasof.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["metrics"]


LGBM = (load("lightgbm") or {}).get("pr_auc")


def baseline(ax):
    if LGBM is not None:
        ax.axhline(LGBM, ls="--", c=GRAY, lw=1.3)
        ax.text(0.99, LGBM, f" LightGBM {LGBM:.3f}", color=GRAY, va="bottom", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=9)


def bars(ax, labels, vals, colors, title, ylabel="test PR-AUC"):
    xs = range(len(vals))
    ax.bar(xs, vals, width=0.55, color=colors, alpha=0.85)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title)


def fig_scaling():
    pts = [("nano_bucket_dt", 2.9), ("mini_bucket_dt", 7.5), ("small_bucket_dt_6k", 13.7)]
    xs, ys = [], []
    for name, p in pts:
        m = load(name)
        if m: xs.append(p); ys.append(m["pr_auc"])
    if len(ys) < 2: return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=8)
    for x, y in zip(xs, ys): ax.annotate(f"{y:.3f}", (x, y), (x, y + 0.02), fontsize=9, ha="center")
    baseline(ax)
    ax.set_xlabel("model size (M parameters)"); ax.set_ylabel("test PR-AUC")
    ax.set_title("(1) Scaling with model size  (as-of-date eval)")
    ax.set_ylim(0, 0.9)
    ax.text(0.02, 0.02, "method (a); subsample base rate ~1.9%; small=6k steps, nano/mini=3k",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig1_model_scaling.png"); plt.close(fig)


def fig_seqlen():
    # eval-window sweep on the L=256 model: short windows fine, long window collapses
    # (RoPE-on-raw-time aliasing over long spans). Files: L256 @ w64 / w128 / native(256).
    pts = [(64, "small_bucket_dt_6k_L256_w64"), (128, "small_bucket_dt_6k_L256_w128"),
           (256, "small_bucket_dt_6k_L256")]
    Ws, ys = [], []
    for w, name in pts:
        m = load(name)
        if m: Ws.append(w); ys.append(m["pr_auc"])
    if len(ys) < 2: return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(Ws, ys, "-o", color=BLUE, lw=2, ms=8)
    for w, y in zip(Ws, ys): ax.annotate(f"{y:.3f}", (w, y), (w, y + 0.03), fontsize=9, ha="center")
    if 256 in Ws:
        ax.annotate("collapse\n(RoPE time-aliasing\nover long spans)", (256, ys[Ws.index(256)]),
                    (150, 0.45), fontsize=8.5, color=GRAY, ha="center",
                    arrowprops=dict(arrowstyle="->", color=GRAY))
    baseline(ax)
    ax.set_xscale("log", base=2); ax.set_xticks(Ws); ax.set_xticklabels(Ws)
    ax.set_xlabel("eval context window  (events, log scale)"); ax.set_ylabel("test PR-AUC")
    ax.set_title("(2) Effect of eval window length  (small L=256 model)")
    ax.set_ylim(0, 0.9)
    ax.text(0.02, 0.02, "method (a); same backbone, only the scoring window varies",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig2_seq_length.png"); plt.close(fig)


def fig_deltat():
    a, b = load("small_bucket_6k_nodt"), load("small_bucket_dt_6k")
    if not (a and b): return
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars(ax, ["no Δt", "+ Δt"], [a["pr_auc"], b["pr_auc"]], [GRAY, BLUE],
         "(3) Δt (time-since-last) embedding  (small, as-of-date)")
    baseline(ax); ax.set_ylim(0, 0.9)
    ax.text(0.02, 0.02, f"method (a); Δt effect ≈ {b['pr_auc']-a['pr_auc']:+.3f} (marginal)",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig3_delta_t.png"); plt.close(fig)


def fig_rope():
    a, b = load("small_bucket_dt_6k_posnone"), load("small_bucket_dt_6k")
    if not (a and b): return
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars(ax, ["no RoPE", "+ RoPE (time)"], [a["pr_auc"], b["pr_auc"]], [GRAY, BLUE],
         "(4) RoPE positional encoding  (small, as-of-date)")
    baseline(ax); ax.set_ylim(0, 0.9)
    ax.text(0.02, 0.02, f"method (a); RoPE effect ≈ {b['pr_auc']-a['pr_auc']:+.3f}",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig4_rope.png"); plt.close(fig)


if __name__ == "__main__":
    fig_scaling(); fig_seqlen(); fig_deltat(); fig_rope()
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p}  ({p.stat().st_size//1024} KB)")
