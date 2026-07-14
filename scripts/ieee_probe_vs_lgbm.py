"""Moonshot: does the frozen FFM (MLM-pretrained, linear probe) beat a strong LightGBM on
IEEE-CIS fraud? Both arms score the SAME stratified as-of-date target subsample so PR-AUC is
directly comparable.

  FFM arm : frozen backbone -> last-position (as-of-date) embedding -> logistic probe.
  LGBM arm: gradient-boosted trees on causal tabular features of the same target rows —
            current-transaction fields (categorical) + within-card causal aggregates.

Run (after pretraining an IEEE checkpoint):
  python scripts/ieee_probe_vs_lgbm.py --ckpt artifacts/pretrain_nano_bucket_ieee.pt \
    --data-dir data/processed_ieee --tokenizer artifacts/tokenizer_ieee.json \
    --parquet data/processed_ieee/transactions.parquet
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from pragma.data.dataset import AsOfDateDataset
from pragma.model.tokenizer import Tokenizer
from pragma.train.probe import load_backbone
from pragma.train.asof_probe import build_targets, embed
from pragma.utils import get_device, seed_everything

CAT = ["ProductCD", "card4", "card6", "card2", "card3", "card5",
       "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType"]


def causal_tabular(parquet: str) -> tuple[pd.DataFrame, list[str]]:
    """(N, features) causal tabular features in global (seq_id, ts) row order == encoded.npz.

    Only past-or-present info per card1 sequence — leakage-safe, mirrors the FFM's as-of-date
    framing. Categorical codes are integer-encoded for LGBM; aggregates are causal (shifted).
    """
    df = pd.read_parquet(parquet)
    g = df.groupby("seq_id", sort=False)
    slog = np.sign(df["amount"]) * np.log1p(np.abs(df["amount"]))
    feats = {
        "amount": df["amount"].to_numpy(np.float32),
        "slog_amount": slog.to_numpy(np.float32),
        "pos_in_seq": g.cumcount().to_numpy(np.float32),                      # history length so far
        "dt_last": g["ts"].diff().fillna(-1).to_numpy(np.float32),            # secs since prev (−1 = first)
        "hour": (pd.to_datetime(df["ts"], unit="s").dt.hour).to_numpy(np.int16),
        "dow": (pd.to_datetime(df["ts"], unit="s").dt.dayofweek).to_numpy(np.int16),
        # causal expanding amount mean (shift so the current row is excluded)
        "amt_mean_prior": g["amount"].apply(lambda s: s.shift().expanding().mean())
                            .reset_index(level=0, drop=True).fillna(-1).to_numpy(np.float32),
    }
    cat_cols = []
    for c in CAT:
        codes = pd.factorize(df[c])[0].astype(np.int32)      # NA -> -1 naturally
        feats[c] = codes
        cat_cols.append(c)
    X = pd.DataFrame(feats)
    return X, cat_cols


def fit_probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    return average_precision_score(yte, s), roc_auc_score(yte, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed_ieee")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer_ieee.json")
    ap.add_argument("--parquet", default="data/processed_ieee/transactions.parquet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-neg", type=int, default=150000)
    ap.add_argument("--out", default="artifacts/ieee_probe_vs_lgbm.json")
    args = ap.parse_args()

    t0 = time.time()
    device = get_device(args.device); seed_everything(0)
    tok = Tokenizer.load(args.tokenizer)
    model, preset, mcfg = load_backbone(args.ckpt, tok, device)
    L = mcfg.max_seq_len
    rng = np.random.default_rng(0)
    tr = build_targets(args.data_dir, "train", rng, args.max_neg)
    te = build_targets(args.data_dir, "test", rng, args.max_neg)

    # --- FFM embedding arm ---
    Etr, ytr = embed(model, AsOfDateDataset(args.data_dir, tr, L), device, args.batch_size)
    Ete, yte = embed(model, AsOfDateDataset(args.data_dir, te, L), device, args.batch_size)
    pr_f, roc_f = fit_probe(Etr, ytr, Ete, yte)
    print(f"[moonshot] FFM embedded ({time.time()-t0:.0f}s)  n_tr={len(ytr):,} n_te={len(yte):,} "
          f"base={yte.mean():.4f}")

    # --- LGBM tabular arm (same target rows) ---
    X, cat_cols = causal_tabular(args.parquet)
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    import lightgbm as lgb
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    d = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, free_raw_data=False)
    booster = lgb.train(dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=128,
                             min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                             bagging_freq=1, scale_pos_weight=spw, seed=0, verbosity=-1), d, 400)
    s = booster.predict(Xte)
    pr_l, roc_l = average_precision_score(yte, s), roc_auc_score(yte, s)

    res = {"dataset": "ieee_cis", "ckpt": args.ckpt, "L": L, "n": int(len(yte)),
           "base_rate": float(yte.mean()),
           "ffm_probe": {"pr_auc": pr_f, "roc_auc": roc_f},
           "lightgbm": {"pr_auc": pr_l, "roc_auc": roc_l}}
    print("\n[moonshot] RESULTS (IEEE-CIS, as-of-date test subsample):")
    print(f"    FFM frozen probe   PR-AUC {pr_f:.4f}  ROC {roc_f:.4f}")
    print(f"    LightGBM baseline  PR-AUC {pr_l:.4f}  ROC {roc_l:.4f}")
    print(f"    delta (FFM-LGBM)   PR-AUC {pr_f-pr_l:+.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[moonshot] wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
