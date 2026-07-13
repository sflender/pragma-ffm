"""Parse the IEEE-CIS (Vesta) fraud CSVs into the canonical parquet the FFM pipeline eats.

Mirrors `parse.py` (TabFormer) but for IEEE-CIS. Emits one row per transaction with:
  seq_id, ts, event_idx, is_fraud, split, amount, <categorical fields…>, and the raw
  relational entity columns (card1 = the sequence entity; addr1 / P_emaildomain / card2 =
  cross-sequence entities for later relational features).

Sequence entity: **card1** (mean ~44 txns/card; cards with ≥5 txns cover 97.7% of rows),
ordered by TransactionDT. Timestamp: TransactionDT is a seconds-offset from a fixed
reference (Kaggle community consensus: 2017-12-01), so ts = REF + TransactionDT gives real
hour/day-of-week for the calendar fields.

Field schema lives in `pragma.data.schema.IEEE_CIS_FIELDS`; encode with `--dataset ieee_cis`.

Run:
  python -m pragma.data.ieee_cis --raw-dir data/raw/ieee_cis --out data/processed_ieee/transactions.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REF_EPOCH = 1_512_086_400  # 2017-12-01 00:00:00 UTC — TransactionDT origin (community consensus)
NA_TOKEN = "<NA>"

# transaction columns we keep (the interpretable core; V/C/D/M engineered cols dropped for v1)
TXN_CAT = ["ProductCD", "card4", "card6", "card2", "card3", "card5",
           "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
ID_CAT = ["DeviceType", "DeviceInfo"]      # from the identity table (join on TransactionID)
SEQ_ENTITY = "card1"


def _as_cat(s: pd.Series) -> pd.Series:
    """Numeric-or-string categorical -> clean string with an explicit NA sentinel.

    Float codes (card2=150.0) become '150', not '150.0'; genuine strings pass through.
    """
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
        out = s.map(lambda x: NA_TOKEN if pd.isna(x) else str(int(x)))
    else:
        out = s.astype("object").where(s.notna(), NA_TOKEN).astype(str)
    return out


def parse(raw_dir: str, out: str, train_q: float, val_q: float,
          split_mode: str = "seq", seed: int = 0) -> None:
    t0 = time.time()
    raw = Path(raw_dir)
    print(f"[ieee] reading {raw}/train_transaction.csv ...")
    keep = (["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", SEQ_ENTITY]
            + TXN_CAT)
    df = pd.read_csv(raw / "train_transaction.csv", usecols=keep)
    print(f"[ieee] {len(df):,} transactions in {time.time()-t0:.1f}s")

    # identity join (DeviceType/Info); most transactions have no identity row -> NA
    idf = pd.read_csv(raw / "train_identity.csv", usecols=["TransactionID"] + ID_CAT)
    df = df.merge(idf, on="TransactionID", how="left")
    print(f"[ieee] joined identity ({df[ID_CAT[0]].notna().sum():,} rows have device info)")

    # --- clean / type ---
    df["amount"] = df["TransactionAmt"].astype("float32")
    df["ts"] = (REF_EPOCH + df["TransactionDT"].astype("int64")).astype("int64")
    df["is_fraud"] = df["isFraud"].astype("int8")
    for c in TXN_CAT + ID_CAT:
        df[c] = _as_cat(df[c])
    df[SEQ_ENTITY] = df[SEQ_ENTITY].astype("int32")

    # --- sequence ids + ordering (one sequence per card1) ---
    df["seq_id"] = df[SEQ_ENTITY].astype("category").cat.codes.astype("int32")
    df = df.sort_values(["seq_id", "ts", "TransactionID"], kind="stable").reset_index(drop=True)
    df["event_idx"] = df.groupby("seq_id").cumcount().astype("int32")

    # --- leakage-safe split (same policy as TabFormer parse.py) ---
    if split_mode == "seq":
        uniq = np.unique(df["seq_id"].to_numpy())
        rng = np.random.default_rng(seed)
        perm = rng.permutation(uniq.size)
        n_tr, n_va = int(train_q * uniq.size), int(val_q * uniq.size)
        code = np.empty(uniq.size, dtype=object)
        code[perm[:n_tr]] = "train"
        code[perm[n_tr:n_va]] = "val"
        code[perm[n_va:]] = "test"
        df["split"] = df["seq_id"].map(dict(zip(uniq.tolist(), code.tolist())))
        print(f"[ieee] split=seq seed={seed} "
              f"train/val/test seqs = {n_tr}/{n_va-n_tr}/{uniq.size-n_va}")
        cut = {}
    else:
        t_train = df["ts"].quantile(train_q)
        t_val = df["ts"].quantile(val_q)
        df["split"] = np.where(df["ts"] <= t_train, "train",
                               np.where(df["ts"] <= t_val, "val", "test"))
        cut = {"train_cutoff_ts": int(t_train), "val_cutoff_ts": int(t_val)}

    keep_cols = (["seq_id", "ts", "event_idx", "amount", "is_fraud", "split", SEQ_ENTITY]
                 + TXN_CAT + ID_CAT)
    df = df[keep_cols]

    # --- report ---
    rep = df.groupby("split").agg(n=("is_fraud", "size"), n_fraud=("is_fraud", "sum"),
                                  seqs=("seq_id", "nunique"))
    rep["fraud_pct"] = 100 * rep["n_fraud"] / rep["n"]
    print(rep.to_string())
    seqlen = df.groupby("seq_id").size()
    print(f"[ieee] seq length: mean={seqlen.mean():.0f} median={seqlen.median():.0f} "
          f"p95={seqlen.quantile(0.95):.0f} max={seqlen.max()}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[ieee] wrote {out}  ({len(df):,} rows, {Path(out).stat().st_size/1e6:.0f} MB) "
          f"in {time.time()-t0:.1f}s")

    meta = {"dataset": "ieee_cis", "split_mode": split_mode, "seq_entity": SEQ_ENTITY,
            "ref_epoch": REF_EPOCH, "n_rows": int(len(df)), "n_seqs": int(df["seq_id"].nunique()),
            "fraud_rate": float(df["is_fraud"].mean()),
            "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()}}
    Path(out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/ieee_cis")
    ap.add_argument("--out", default="data/processed_ieee/transactions.parquet")
    ap.add_argument("--train-q", type=float, default=0.80)
    ap.add_argument("--val-q", type=float, default=0.90)
    ap.add_argument("--split-mode", choices=["seq", "temporal"], default="seq")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    parse(args.raw_dir, args.out, args.train_q, args.val_q, args.split_mode, args.seed)


if __name__ == "__main__":
    main()
