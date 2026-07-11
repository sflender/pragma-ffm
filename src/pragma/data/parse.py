"""Parse the raw IBM TabFormer credit-card CSV into a clean, typed parquet.

Responsibilities (kept deliberately minimal — encoding/tokenisation lives in the
tokenizer, not here):
  * parse the Year/Month/Day/Time columns into a single UTC timestamp,
  * clean the ``Amount`` string ("$134.09" / "$-1.00") into a float,
  * fill nulls in optional categoricals with an explicit "NA" sentinel,
  * assign a per-(User, Card) integer ``seq_id`` and sort each sequence by time,
  * assign a leakage-safe temporal ``split`` (train/val/test) via global timestamp
    quantile cutoffs.

The output has one row per transaction and preserves raw string categoricals so the
tokenizer can fit its vocab / hashing / bucketing on the TRAIN split only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DEFAULT = "data/raw/TabFormer/data/credit_card/card_transaction.v1.csv"
OUT_DEFAULT = "data/processed/transactions.parquet"

# Categoricals that have nulls (online txns have no state/zip; Errors? usually empty).
NA_TOKEN = "<NA>"
CAT_FILL = {
    "Merchant State": NA_TOKEN,
    "Zip": NA_TOKEN,
    "Errors?": NA_TOKEN,
}


def clean_amount(s: pd.Series) -> pd.Series:
    """'$134.09' / '$-1.00' -> float32."""
    return (
        s.str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype("float32")
    )


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Combine Year/Month/Day/Time into a single datetime, then epoch seconds (int64)."""
    dt = pd.to_datetime(
        dict(
            year=df["Year"].astype(int),
            month=df["Month"].astype(int),
            day=df["Day"].astype(int),
        )
    )
    # Time is "HH:MM"; add as a timedelta.
    hm = df["Time"].str.split(":", expand=True).astype(int)
    dt = dt + pd.to_timedelta(hm[0], unit="h") + pd.to_timedelta(hm[1], unit="m")
    return (dt.values.astype("int64") // 1_000_000_000).astype("int64")  # ns -> s


def parse(raw: str, out: str, train_q: float, val_q: float,
          split_mode: str = "seq", seed: int = 0) -> None:
    t0 = time.time()
    print(f"[parse] reading {raw} ...")
    df = pd.read_csv(raw, dtype=str)
    print(f"[parse] {len(df):,} rows in {time.time() - t0:.1f}s")

    # --- clean / type ---
    df["amount"] = clean_amount(df["Amount"])
    df["ts"] = build_timestamp(df)
    df["is_fraud"] = (df["Is Fraud?"] == "Yes").astype("int8")

    for col, fill in CAT_FILL.items():
        df[col] = df[col].fillna(fill)

    rename = {
        "User": "user", "Card": "card", "Use Chip": "use_chip",
        "Merchant Name": "merchant_name", "Merchant City": "merchant_city",
        "Merchant State": "merchant_state", "Zip": "zip", "MCC": "mcc",
        "Errors?": "errors",
    }
    df = df.rename(columns=rename)

    keep = [
        "user", "card", "ts", "amount", "use_chip", "merchant_name",
        "merchant_city", "merchant_state", "zip", "mcc", "errors", "is_fraud",
    ]
    df = df[keep]

    # --- sequence ids + ordering ---
    df["user"] = df["user"].astype("int32")
    df["card"] = df["card"].astype("int32")
    seq_key = df["user"].astype("int64") * 1000 + df["card"].astype("int64")
    df["seq_id"] = seq_key.astype("category").cat.codes.astype("int32")
    df = df.sort_values(["seq_id", "ts"], kind="stable").reset_index(drop=True)
    df["event_idx"] = df.groupby("seq_id").cumcount().astype("int32")

    # --- leakage-safe split ---
    # "seq" : whole (user,card) sequences assigned to a split (default; clean test of
    #          generalisation to unseen users, standard in event-sequence FM work).
    # "temporal": global timestamp quantile cutoffs (realistic but shift-confounded here).
    if split_mode == "seq":
        seqs = df["seq_id"].to_numpy()
        uniq = np.unique(seqs)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(uniq.size)
        n_tr, n_va = int(train_q * uniq.size), int(val_q * uniq.size)
        code = np.empty(uniq.size, dtype=object)
        code[perm[:n_tr]] = "train"
        code[perm[n_tr:n_va]] = "val"
        code[perm[n_va:]] = "test"
        seq_to_split = dict(zip(uniq.tolist(), code.tolist()))
        df["split"] = df["seq_id"].map(seq_to_split)
        print(f"[parse] split=seq  seed={seed}  "
              f"train/val/test seqs = {n_tr}/{n_va - n_tr}/{uniq.size - n_va}")
        cut = {}
    else:
        t_train = df["ts"].quantile(train_q)
        t_val = df["ts"].quantile(val_q)
        df["split"] = np.where(df["ts"] <= t_train, "train",
                               np.where(df["ts"] <= t_val, "val", "test"))
        print(f"[parse] split=temporal cutoffs: train<= {pd.to_datetime(t_train, unit='s').date()} "
              f"val<= {pd.to_datetime(t_val, unit='s').date()}")
        cut = {"train_cutoff_ts": int(t_train), "val_cutoff_ts": int(t_val)}

    # --- report ---
    rep = df.groupby("split").agg(
        n=("is_fraud", "size"),
        n_fraud=("is_fraud", "sum"),
        seqs=("seq_id", "nunique"),
    )
    rep["fraud_pct"] = 100 * rep["n_fraud"] / rep["n"]
    print(rep.to_string())

    seqlen = df.groupby("seq_id").size()
    print(f"[parse] seq length: mean={seqlen.mean():.0f} median={seqlen.median():.0f} "
          f"p95={seqlen.quantile(0.95):.0f} max={seqlen.max()}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[parse] wrote {out}  ({len(df):,} rows, {Path(out).stat().st_size/1e6:.0f} MB) "
          f"in {time.time() - t0:.1f}s total")

    meta = {
        "split_mode": split_mode,
        "n_rows": int(len(df)),
        "n_seqs": int(df["seq_id"].nunique()),
        "n_users": int(df["user"].nunique()),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
        **cut,
    }
    Path(out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--train-q", type=float, default=0.80)
    ap.add_argument("--val-q", type=float, default=0.90)
    ap.add_argument("--split-mode", choices=["seq", "temporal"], default="seq")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    parse(args.raw, args.out, args.train_q, args.val_q, args.split_mode, args.seed)


if __name__ == "__main__":
    main()
