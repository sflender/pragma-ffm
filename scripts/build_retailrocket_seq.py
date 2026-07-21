"""Retailrocket e-commerce clickstream -> canonical parquet (NON-FRAUD conversion row).

SEQUENCE entity = visitorid; SHARED cross-visitor entity = itemid (an item trends across many
visitors concurrently). Label is_fraud is reused for is_conversion = event in {addtocart,
transaction} (option 2: purchase-intent, ~3% base -- less sparse than purchase-only).

The event TYPE is NOT emitted as a field (it is the label). The encoder sees a capped item bucket
(top-N items + OOV, since raw itemid is ~235k) + calendar (hour/dow, derived from ts). The FULL
factorized item is kept as `item_id` for build_merchant_memory / build_entity_neighbors --entity item_id.

Run: python scripts/build_retailrocket_seq.py --csv events.csv --out data/rr_seq/transactions.parquet
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Retailrocket events.csv")
    ap.add_argument("--out", default="data/rr_seq/transactions.parquet")
    ap.add_argument("--item-vocab", type=int, default=10000, help="top-N items kept as buckets; rest -> OOV(0)")
    ap.add_argument("--nrows", type=int, default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.csv, nrows=a.nrows, usecols=["timestamp", "visitorid", "event", "itemid"])
    df["ts"] = (df["timestamp"].astype("int64") // 1000).astype(np.int64)     # ms -> s
    df["seq_id"] = pd.factorize(df["visitorid"])[0].astype(np.int32)
    df["item_id"] = pd.factorize(df["itemid"])[0].astype(np.int32)            # full item (shared entity)
    # capped item bucket for the encoder field: top-N items -> 1..N, rest -> 0 (OOV)
    top = df["itemid"].value_counts().index[: a.item_vocab]
    code = {it: i + 1 for i, it in enumerate(top)}
    df["item"] = df["itemid"].map(code).fillna(0).astype(np.int32)
    df["is_fraud"] = df["event"].isin(["addtocart", "transaction"]).astype(np.int8)   # is_conversion
    df["amount"] = np.float32(0.0)                                            # no numeric field

    df = df.sort_values(["seq_id", "ts"], kind="stable").reset_index(drop=True)
    df["event_idx"] = df.groupby("seq_id").cumcount().astype(np.int32)
    ts = df["ts"].to_numpy()
    q75, q80 = np.quantile(ts, 0.75), np.quantile(ts, 0.80)
    df["split"] = np.where(ts > q80, "test", np.where(ts > q75, "val", "train"))

    cols = ["seq_id", "ts", "event_idx", "is_fraud", "split", "amount", "item", "item_id"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_parquet(a.out, index=False)
    rep = df.groupby("split").agg(n=("is_fraud", "size"), f=("is_fraud", "sum"))
    print(f"[rr] wrote {a.out}  {len(df):,} rows  visitors(seq)={df['seq_id'].nunique():,}  "
          f"items={df['item_id'].nunique():,}  conv {df['is_fraud'].mean():.4f}")
    print(rep.to_string())
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(
        {"dataset": "retailrocket", "n_rows": int(len(df)), "conv_rate": float(df["is_fraud"].mean())}, indent=2))


if __name__ == "__main__":
    main()
