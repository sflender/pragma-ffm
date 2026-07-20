"""Matched LightGBM baseline for the synthetic modes.

Fair comparison to the FFM: LightGBM gets the per-transaction fields the event encoder sees
(amount, mcc, hour, dow) PLUS the exact causal merchant-memory features the FFM's memory path
sees (merchant_mem.npz: prior count, mean amount, recency, windowed velocities). It does NOT get
the raw merchant id -- that leaks a label-derived per-merchant fraud rate (the 0.605 artefact).
Evaluated on the SAME build_targets test subsample (seed 0) the FFM reports on, so PR-AUC is
apples-to-apples.

Run: python scripts/synth_lgbm_matched.py --data-dir data/synth_pattern --out artifacts/synth_pattern_lgbm.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from pragma.train.asof_probe import build_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-neg", type=int, default=150000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(f"{a.data_dir}/transactions.parquet")
    ts = df["ts"].to_numpy(np.int64)
    hour = ((ts // 3600) % 24).astype(np.float32)
    dow = ((ts // 86400) % 7).astype(np.float32)
    base = np.column_stack([df["amount"].to_numpy(np.float32),
                            df["mcc"].astype(int).to_numpy(np.float32), hour, dow])
    memp = Path(a.data_dir) / "merchant_mem.npz"
    if memp.exists():
        mem = np.load(memp)["mem"].astype(np.float32)
        X = np.concatenate([base, mem], axis=1)
        feat_names = ["amount", "mcc", "hour", "dow"] + [f"mem{i}" for i in range(mem.shape[1])]
    else:
        X = base; feat_names = ["amount", "mcc", "hour", "dow"]
    y = np.load(f"{a.data_dir}/encoded.npz")["is_fraud"].astype(np.int8)

    tr = build_targets(a.data_dir, "train", np.random.default_rng(0), a.max_neg)
    te = build_targets(a.data_dir, "test", np.random.default_rng(0), a.max_neg)
    spw = (y[tr] == 0).sum() / max(1, int(y[tr].sum()))
    clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=64, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                             min_child_samples=100, random_state=0, verbose=-1)
    clf.fit(X[tr], y[tr], feature_name=feat_names)
    s = clf.predict_proba(X[te])[:, 1]
    res = {"data_dir": a.data_dir, "n_test": int(te.size), "base_rate": float(y[te].mean()),
           "pr_auc": float(average_precision_score(y[te], s)),
           "roc_auc": float(roc_auc_score(y[te], s)),
           "n_features": X.shape[1]}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[lgbm] {a.data_dir}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
          f"(base {res['base_rate']:.4f}, n_test {res['n_test']:,}, {X.shape[1]} feats)")


if __name__ == "__main__":
    main()
