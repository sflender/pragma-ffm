"""Fusion probe: frozen FFM record embedding  (+)  causal relational (merchant) features.

Tests whether cross-entity signal the per-sequence backbone can't see (a merchant's recent
behaviour across ALL cards) complements the FFM embedding for fraud. Same stratified
sliding-window as-of-date subsample as ``asof_probe`` (n=152,928, base 0.0191), so the
embedding-only arm reproduces the standard probe number as a control.

Reports PR-AUC / ROC-AUC for: embedding-only, relational-only, and fusion (concat), with a
LightGBM fusion as a nonlinear bonus.

Run (on a box with the checkpoint + parquet):
  python scripts/fusion_probe.py --ckpt artifacts/pretrain_small_bucket_s15000.pt \
    --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json \
    --parquet data/processed/transactions.parquet
"""
from __future__ import annotations

import argparse, json, time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from pragma.data.dataset import AsOfDateDataset
from pragma.model.tokenizer import Tokenizer
from pragma.train.probe import load_backbone
from pragma.train.asof_probe import build_targets, embed
from pragma.utils import get_device, seed_everything

REL = ["m_pop", "m_prior_fraud_rate", "cm_new"]


def relational_features(parquet: str) -> np.ndarray:
    """(N,3) causal merchant features in global (seq_id,ts) row order == encoded.npz order."""
    df = pd.read_parquet(parquet, columns=["ts", "is_fraud", "card", "merchant_name"])
    N = len(df)
    o = pd.DataFrame({"gidx": np.arange(N, dtype=np.int64),
                      "merch": pd.factorize(df["merchant_name"])[0].astype(np.int32),
                      "card": df["card"].to_numpy(), "ts": df["ts"].to_numpy(),
                      "is_fraud": df["is_fraud"].to_numpy(np.int8)})
    o = o.sort_values(["merch", "ts"], kind="stable")
    g = o.groupby("merch", sort=False)
    mpt = g.cumcount().to_numpy(np.int64)                              # prior txns at merchant
    mpf = (g["is_fraud"].cumsum().to_numpy() - o["is_fraud"].to_numpy())   # causal prior fraud count
    mpr = (mpf / np.clip(mpt, 1, None)).astype(np.float32)            # prior fraud rate (label-based)
    o["cm"] = o["card"].astype(np.int64) * 100_000_000 + o["merch"]
    cmnew = (o.groupby("cm", sort=False).cumcount().to_numpy() == 0).astype(np.float32)
    o["f0"] = np.log1p(mpt).astype(np.float32); o["f1"] = mpr; o["f2"] = cmnew
    o = o.sort_values("gidx", kind="stable")
    return np.column_stack([o["f0"].to_numpy(), o["f1"].to_numpy(), o["f2"].to_numpy()])


def fit_lr(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    return average_precision_score(yte, s), roc_auc_score(yte, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed_dt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer_dt.json")
    ap.add_argument("--parquet", default="data/processed/transactions.parquet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="artifacts/eval_fusion.json")
    args = ap.parse_args()

    t0 = time.time()
    device = get_device(args.device); seed_everything(0)
    tok = Tokenizer.load(args.tokenizer)
    model, preset, mcfg = load_backbone(args.ckpt, tok, device)
    L = mcfg.max_seq_len
    rng = np.random.default_rng(0)                         # SAME order as asof_probe.run
    tr = build_targets(args.data_dir, "train", rng, 150000)
    te = build_targets(args.data_dir, "test", rng, 150000)

    Etr, ytr = embed(model, AsOfDateDataset(args.data_dir, tr, L), device, args.batch_size)
    Ete, yte = embed(model, AsOfDateDataset(args.data_dir, te, L), device, args.batch_size)
    print(f"[fusion] embedded train {len(ytr):,} test {len(yte):,} in {time.time()-t0:.0f}s")

    rel = relational_features(args.parquet)
    Rtr, Rte = rel[tr], rel[te]
    print(f"[fusion] relational features built ({time.time()-t0:.0f}s)")

    pr_e, roc_e = fit_lr(Etr, ytr, Ete, yte)
    pr_r, roc_r = fit_lr(Rtr, ytr, Rte, yte)
    pr_f, roc_f = fit_lr(np.hstack([Etr, Rtr]), ytr, np.hstack([Ete, Rte]), yte)

    # nonlinear fusion bonus (LightGBM on emb+rel)
    try:
        import lightgbm as lgb
        spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
        d = lgb.Dataset(np.hstack([Etr, Rtr]), ytr, free_raw_data=False)
        m = lgb.train(dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=128,
                           min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                           bagging_freq=1, scale_pos_weight=spw, seed=0, verbosity=-1), d, 400)
        s = m.predict(np.hstack([Ete, Rte]))
        pr_g, roc_g = average_precision_score(yte, s), roc_auc_score(yte, s)
    except Exception as e:
        pr_g, roc_g = -1, -1; print("[fusion] lgbm bonus failed:", e)

    res = {"ckpt": args.ckpt, "L": L, "n": int(len(yte)), "base": float(yte.mean()),
           "embedding_only": {"pr_auc": pr_e, "roc_auc": roc_e},
           "relational_only": {"pr_auc": pr_r, "roc_auc": roc_r},
           "fusion_logreg": {"pr_auc": pr_f, "roc_auc": roc_f},
           "fusion_lgbm": {"pr_auc": pr_g, "roc_auc": roc_g}}
    print("[fusion] RESULTS:")
    print(f"    embedding-only    PR-AUC {pr_e:.3f}  ROC {roc_e:.3f}   (control ~= asof_probe)")
    print(f"    relational-only   PR-AUC {pr_r:.3f}  ROC {roc_r:.3f}")
    print(f"    fusion (logreg)   PR-AUC {pr_f:.3f}  ROC {roc_f:.3f}   (lift {pr_f-pr_e:+.3f})")
    print(f"    fusion (lgbm)     PR-AUC {pr_g:.3f}  ROC {roc_g:.3f}   (lift {pr_g-pr_e:+.3f})")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[fusion] wrote {args.out}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
