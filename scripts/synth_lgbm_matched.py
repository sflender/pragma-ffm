"""Matched LightGBM baseline for the synthetic modes — a SINGLE strong tabular arm.

LightGBM gets the per-transaction fields the FFM's event encoder sees (amount, mcc, hour, dow)
PLUS the standard causal aggregates a practitioner engineers for BOTH entities (the card and the
merchant): prior count, windowed velocity (last 1h / 15m), expanding mean amount, and recency
(seconds since the entity's previous txn). All aggregates are causal (shifted, no leakage). It does
NOT get raw card/merchant ids (which would leak the label-derived per-merchant fraud rate / overfit)
and NOT a bespoke "MCC vs the merchant's usual MCC" detector (that is precisely the content pattern
the cross-sequence FFM is meant to discover from raw neighbours).

Evaluated on the SAME build_targets test subsample (seed 0) the FFM reports on -> apples-to-apples.

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


def _entity_aggs(df, key, prefix, windows=(3600, 900)):
    """Causal per-entity features, aligned to df's current (seq_id, ts) row order."""
    g = df.groupby(key, sort=False)
    out = {}
    out[f"{prefix}_count"] = g.cumcount().to_numpy(np.float32)
    prev_ts = g["ts"].shift()
    out[f"{prefix}_recency"] = np.log1p(np.clip((df["ts"] - prev_ts).to_numpy(), 0, None)).astype(np.float32)
    out[f"{prefix}_meanamt"] = g["amount"].apply(lambda s: s.shift().expanding().mean()).to_numpy(np.float32)
    # windowed velocity: # prior same-entity txns within W seconds (burst detector)
    ts = df["ts"].to_numpy(np.int64)
    order = df.index.to_numpy()
    for W in windows:
        vel = np.zeros(len(df), np.float32)
        for _, idx in g.indices.items():
            t = ts[idx]
            vel[idx] = np.arange(len(t)) - np.searchsorted(t, t - W, side="left")
        out[f"{prefix}_vel{W}"] = np.log1p(vel).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-neg", type=int, default=150000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(f"{a.data_dir}/transactions.parquet")            # encoded.npz row order
    hour = ((df["ts"] // 3600) % 24).astype(np.float32)
    dow = ((df["ts"] // 86400) % 7).astype(np.float32)
    cols = {"amount": df["amount"].to_numpy(np.float32),
            "mcc": df["mcc"].astype(int).to_numpy(np.float32),
            "hour": hour.to_numpy(), "dow": dow.to_numpy()}
    mats, feat_names = [np.column_stack(list(cols.values()))], list(cols.keys())
    # merchant: reuse the tested build_merchant_memory features (the exact windowed velocity the FFM
    # sees), row-aligned via merch_mem.npz. card: computed inline (build_merchant_memory can't take
    # entity==card-col). Both give the tabular baseline per-card AND per-merchant causal aggregates.
    mp = Path(a.data_dir) / "merch_mem.npz"
    if mp.exists():
        m = np.load(mp)["mem"].astype(np.float32)
        mats.append(m); feat_names += [f"merch{i}" for i in range(m.shape[1])]
    else:
        ag = _entity_aggs(df.reset_index(drop=True), "merchant", "merch")
        mats.append(ag.to_numpy(np.float32)); feat_names += list(ag.columns)
    cag = _entity_aggs(df.reset_index(drop=True), "card", "card")
    mats.append(cag.to_numpy(np.float32)); feat_names += list(cag.columns)
    X = np.concatenate(mats, axis=1).astype(np.float32)
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
           "roc_auc": float(roc_auc_score(y[te], s)), "n_features": X.shape[1],
           "top_features": [feat_names[i] for i in np.argsort(clf.feature_importances_)[::-1][:6]]}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[lgbm] {a.data_dir}: PR-AUC={res['pr_auc']:.4f} ROC={res['roc_auc']:.4f} "
          f"(base {res['base_rate']:.4f}, {X.shape[1]} feats)  top={res['top_features'][:4]}")


if __name__ == "__main__":
    main()
