"""Precompute the per-event causal MERCHANT-memory vector for relational cross-attention.

For every transaction, summarise its merchant's recent activity **across all cards**, using
only events strictly before it (causal / leakage-safe). Written aligned to the global
(seq_id, ts) row order — i.e. the same order as encoded.npz — to `{data-dir}/merchant_mem.npz`.

d_mem = 5 features (z-scored by the train split):
  0  log prior txn count at merchant      (popularity)
  1  merchant prior fraud rate            (label-based; strongest signal on TabFormer)
  2  merchant prior mean signed-log-amount(what this merchant usually charges)
  3  log seconds since merchant's prev txn(recency / velocity — small = burst)
  4  card-merchant novelty (first time this card at this merchant)

Run:  python scripts/build_merchant_memory.py --parquet data/processed/transactions.parquet \
          --data-dir data/processed_dt
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/transactions.parquet")
    ap.add_argument("--data-dir", default="data/processed_dt")
    ap.add_argument("--entity", default="merchant_name",
                    help="entity column to aggregate over (e.g. addr1 for IEEE)")
    ap.add_argument("--card-col", default="card",
                    help="the sequence/card column, for the card-entity novelty feature")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.parquet, columns=["ts", "is_fraud", args.card_col, "amount",
                                                args.entity, "split"])
    N = len(df)
    slog_amt = (np.sign(df["amount"].to_numpy()) * np.log1p(np.abs(df["amount"].to_numpy()))).astype(np.float32)
    o = pd.DataFrame({"gidx": np.arange(N, dtype=np.int64),
                      "merch": pd.factorize(df[args.entity])[0].astype(np.int32),
                      "card": df[args.card_col].to_numpy(), "ts": df["ts"].to_numpy(np.int64),
                      "isf": df["is_fraud"].to_numpy(np.int8), "slog": slog_amt})
    o = o.sort_values(["merch", "ts"], kind="stable")
    g = o.groupby("merch", sort=False)
    n_prior = g.cumcount().to_numpy(np.float64)                               # 0-based prior count
    denom = np.clip(n_prior, 1, None)
    prior_fraud = (g["isf"].cumsum().to_numpy() - o["isf"].to_numpy()) / denom
    prior_meanamt = (g["slog"].cumsum().to_numpy() - o["slog"].to_numpy()) / denom
    prev_ts = g["ts"].shift(1).to_numpy()
    dt_merch = np.where(np.isnan(prev_ts), 1e7, o["ts"].to_numpy() - prev_ts)  # first-at-merchant -> large gap
    o["cm"] = o["card"].astype(np.int64) * 100_000_000 + o["merch"]
    cm_new = (o.groupby("cm", sort=False).cumcount().to_numpy() == 0).astype(np.float32)

    feats = np.column_stack([
        np.log1p(n_prior),
        prior_fraud,
        prior_meanamt,
        np.log1p(np.clip(dt_merch, 0, None)),
        cm_new,
    ]).astype(np.float32)
    o_feats = pd.DataFrame(feats, index=o.index)
    o_feats["gidx"] = o["gidx"].to_numpy()
    o_feats = o_feats.sort_values("gidx", kind="stable")               # restore global order
    mem = o_feats.drop(columns="gidx").to_numpy(np.float32)

    # z-score by TRAIN split (mean/std over train rows), applied to all
    tr = df["split"].to_numpy() == "train"
    mu, sd = mem[tr].mean(0), mem[tr].std(0)
    sd[sd < 1e-6] = 1.0
    mem = ((mem - mu) / sd).astype(np.float32)

    out = Path(args.data_dir) / "merchant_mem.npz"
    np.savez(out, mem=mem)
    print(f"[mem] wrote {out}  shape={mem.shape}  in {time.time()-t0:.0f}s")
    print(f"[mem] feature means (post-z): {mem.mean(0).round(3)}  (train)")


if __name__ == "__main__":
    main()
