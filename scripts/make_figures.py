"""Generate all figures from the method-(a) sliding-window as-of-date results (eval_*_swasof.json).

Figures:
  fig1 model scaling · fig2 seq-length · fig3 Δt · fig4 RoPE · fig5 all-ablations (bars)
  fig6 small-6k loss curve (parses a captured pretrain log) · fig7 LightGBM feature importance.

All PR-AUC numbers are on the same stratified test subsample (base rate ~1.9%). Missing files are
skipped gracefully, so this can be run at any time and re-run as new results land. Outputs to figures/.

Run:  python scripts/make_figures.py [--loss-log PATH] [--only fig1,fig5,...]
"""
from __future__ import annotations

import argparse
import json
import re
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
    # each model trained AND scored at its OWN context length L (used as designed)
    pts = [(64, "small_bucket_dt_6k_L64"), (128, "small_bucket_dt_6k"),
           (256, "small_bucket_dt_6k_L256")]
    Ls, ys = [], []
    for L, name in pts:
        m = load(name)
        if m: Ls.append(L); ys.append(m["pr_auc"])
    if len(ys) < 2: return
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(Ls, ys, "-o", color=BLUE, lw=2, ms=8)
    for L, y in zip(Ls, ys): ax.annotate(f"{y:.3f}", (L, y), (L, y + 0.03), fontsize=9, ha="center")
    if 256 in Ls:
        ax.annotate("L=256 collapses at its native window\n(recovers to ~0.81 scored at L≤128;\nlong-window readout artifact — E10)",
                    (256, ys[Ls.index(256)]), (92, 0.44), fontsize=8, color=GRAY, ha="left",
                    arrowprops=dict(arrowstyle="->", color=GRAY))
    baseline(ax)
    ax.set_xscale("log", base=2); ax.set_xticks(Ls); ax.set_xticklabels(Ls)
    ax.set_xlabel("training context length  L  (trained & scored at L)"); ax.set_ylabel("test PR-AUC")
    ax.set_title("(2) Scaling with training context length  (small)")
    ax.set_ylim(0, 0.9)
    ax.text(0.02, 0.02, "method (a); each model trained AND scored at its own L",
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


def _pr(name):
    m = load(name)
    return m["pr_auc"] if m else None


def fig_ablations():
    """(5) All three architecture ablations as grouped bars, ablated-vs-full (shared baseline)."""
    full = _pr("small_bucket_dt_6k")                     # RoPE + Δt + field-emb, all ON
    comps = [("RoPE\n(time)", _pr("small_bucket_dt_6k_posnone")),
             ("Δt\nembed",   _pr("small_bucket_6k_nodt")),
             ("field\nemb",  _pr("small_bucket_dt_6k_nofe"))]
    if full is None or any(v is None for _, v in comps):
        print("[fig5] missing eval json(s); skipped"); return
    labels = [c[0] for c in comps]; abl = [c[1] for c in comps]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - w/2, abl, w, color=GRAY, alpha=0.85, label="ablated (component OFF)")
    ax.bar(x + w/2, [full]*len(x), w, color=BLUE, alpha=0.85, label="full model (ON)")
    for xi, v in zip(x, abl):
        ax.text(xi - w/2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
        ax.text(xi + w/2, full + 0.006, f"{full:.3f}", ha="center", fontsize=9)
        d = full - v
        ax.annotate(f"Δ={d:+.3f}", (xi, max(full, v) + 0.03), ha="center", fontsize=9,
                    color=(BLUE if d > 0.01 else GRAY),
                    fontweight=("bold" if d > 0.01 else "normal"))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("test PR-AUC (method a)"); ax.set_ylim(0.65, 0.90)
    ax.set_title("(5) Architecture ablations  (small, 6k steps, as-of-date)")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.02, "shared full-model baseline 0.786; field-emb Δ within seed noise (E11)",
            transform=ax.transAxes, fontsize=8, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "fig5_ablations.png"); plt.close(fig)


_LOSS_LINE = re.compile(r"step\s+(\d+)/(\d+)\s+loss\s+([\d.]+).*?([\d.]+)\s*it/s")


def fig_loss_curve(log_path="/tmp/small_losscurve.log"):
    """(6) MLM training loss vs step (+ wall-clock axis), parsed from a captured pretrain log."""
    p = Path(log_path)
    if not p.exists():
        print(f"[fig6] loss log {log_path} not found; skipped"); return
    rows = [(int(m.group(1)), float(m.group(3)), float(m.group(4)))
            for m in _LOSS_LINE.finditer(p.read_text())]
    if len(rows) < 5:
        print(f"[fig6] only {len(rows)} step-lines parsed; skipped"); return
    rows.sort()
    steps = np.array([r[0] for r in rows]); loss = np.array([r[1] for r in rows])
    its = np.array([r[2] for r in rows])
    mins = np.cumsum(np.diff(steps, prepend=0) / np.clip(its, 1e-6, None)) / 60.0
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(steps, loss, color=BLUE, lw=1.8)
    ax.set_xlabel("training step"); ax.set_ylabel("MLM loss (train, running mean)")
    ax.set_title("(6) Pretraining loss — small, 6k steps")
    ax.text(0.97, 0.95, f"final loss {loss[-1]:.3f}\n~{mins[-1]:.0f} min wall-clock (MPS)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GRAY,
            bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.8))
    ax.secondary_xaxis("top", functions=(
        lambda s: np.interp(s, steps, mins), lambda t: np.interp(t, mins, steps))
    ).set_xlabel("wall-clock (min, MPS)")
    fig.tight_layout(); fig.savefig(OUT / "fig6_loss_curve.png"); plt.close(fig)


def fig_lgbm_importance():
    """(7) LightGBM gain importance — retrains the same baseline as scripts/lgbm_subsample.py."""
    import lightgbm as lgb
    import pandas as pd
    from pragma.baselines.features import CAT_COLS, build_features
    df = pd.read_parquet("data/processed/transactions.parquet", columns=[
        "seq_id", "ts", "amount", "use_chip", "mcc", "merchant_state", "errors", "is_fraud", "split"])
    X = build_features(df); y = df["is_fraud"].to_numpy(np.int8); split = df["split"].to_numpy()
    tr, va = split == "train", split == "val"
    spw = (~y[tr].astype(bool)).sum() / max(1, int(y[tr].sum()))
    dtrain = lgb.Dataset(X[tr], y[tr], categorical_feature=CAT_COLS, free_raw_data=False)
    dval = lgb.Dataset(X[va], y[va], categorical_feature=CAT_COLS, reference=dtrain)
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=128,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, scale_pos_weight=spw, seed=0, verbosity=-1)
    print("[fig7] training LightGBM (matches lgbm_subsample.py) ...")
    model = lgb.train(params, dtrain, 800, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
    # normalized gain: fraction of total split-gain per feature (ratios are the meaningful part)
    gain = model.feature_importance(importance_type="gain").astype(float)
    names = model.feature_name()
    frac = gain / max(gain.sum(), 1e-9)
    order = np.argsort(frac)                              # ascending -> largest ends up on top
    names = [names[i] for i in order]; frac = frac[order]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(names))
    ax.barh(y, frac, color=BLUE, alpha=0.9)
    for yi, f in zip(y, frac):
        ax.text(f + 0.008, yi, f"{f*100:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0, frac.max() * 1.15)
    ax.set_xlabel("share of total gain"); ax.set_ylabel("feature")
    ax.set_title("(7) LightGBM feature importance  (gain, normalized)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig7_lgbm_importance.png"); plt.close(fig)


FIGS = {
    "fig1": fig_scaling, "fig2": fig_seqlen, "fig3": fig_deltat, "fig4": fig_rope,
    "fig5": fig_ablations, "fig7": fig_lgbm_importance,       # fig6 needs the loss-log path
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-log", default="/tmp/small_losscurve.log")
    ap.add_argument("--only", default="", help="comma list of figN to render (default: all)")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    for name, fn in FIGS.items():
        if not only or name in only:
            fn()
    if not only or "fig6" in only:
        fig_loss_curve(args.loss_log)
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p}  ({p.stat().st_size//1024} KB)")