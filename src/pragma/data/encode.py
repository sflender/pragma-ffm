"""Fit the tokenizer on the train split and pre-encode all events to integer arrays.

Outputs (under data/processed/):
  * ``encoded.npz`` : codes (N,F int16), ts (N int64), seq_id (N int32),
                      is_fraud (N int8), split (N int8: 0=train,1=val,2=test)
  * ``seq_index.npz``: per-sequence contiguous [start,end) offsets into the arrays
                      (rows are already sorted by seq_id then ts by parse.py)
  * ``artifacts/tokenizer.json``

The heavy work runs once; training/eval then memory-maps these arrays.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pragma.config import TokenizerConfig
from pragma.model.tokenizer import Tokenizer

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def build(parquet: str, out_dir: str, tok_path: str,
          n_amount_buckets: int, hash_buckets: int,
          include_dt: bool = False, n_dt_buckets: int = 20,
          dt_min_s: float = 1.0, dt_max_s: float = 31_536_000.0,
          mname_cat: bool = False, zip_prefix: int | None = None) -> None:
    t0 = time.time()
    Path(out_dir).mkdir(parents=True, exist_ok=True)          # create out dir if missing
    Path(tok_path).parent.mkdir(parents=True, exist_ok=True)  # ...and the tokenizer's dir
    df = pd.read_parquet(parquet)
    print(f"[encode] loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    if include_dt:
        # time since previous event in the same sequence (causal per-row). first event -> 0.
        df["dt"] = df.groupby("seq_id")["ts"].diff().fillna(0).clip(lower=0)

    # experiment hooks (tokenization ablations) -----------------------------------------
    kind_overrides: dict = {}
    if mname_cat:                                # (a) merchant_name: hash -> full identity vocab
        kind_overrides["merchant_name"] = "cat"
    if zip_prefix:                               # (b) zip: hash -> N-digit regional prefix as cat
        df["zip"] = df["zip"].astype(str).str[:zip_prefix]
        kind_overrides["zip"] = "cat"
    if kind_overrides:
        print(f"[encode] kind_overrides={kind_overrides}"
              + (f" zip_prefix={zip_prefix}" if zip_prefix else ""))

    tok = Tokenizer.fit(df[df.split == "train"], n_amount_buckets, hash_buckets,
                        include_dt, n_dt_buckets, dt_min_s, dt_max_s,
                        kind_overrides=kind_overrides or None)
    tok.save(tok_path)
    print(f"[encode] tokenizer: F={tok.F} fields, V={tok.V} total vocab -> {tok_path}")
    for f in tok.fields:
        print(f"    {f.name:16s} {f.kind:5s} vocab={f.vocab}")

    codes = tok.encode_frame(df).astype(np.int16)   # vocab per field << 32k, int16 is safe
    ts = df["ts"].to_numpy(np.int64)
    seq_id = df["seq_id"].to_numpy(np.int32)
    amount_raw = df["amount"].to_numpy(np.float32)  # raw value for PLE/periodic numeric modes
    is_fraud = df["is_fraud"].to_numpy(np.int8)
    split = df["split"].map(SPLIT_CODE).to_numpy(np.int8)

    out = Path(out_dir)
    np.savez(out / "encoded.npz", codes=codes, ts=ts, seq_id=seq_id,
             amount_raw=amount_raw, is_fraud=is_fraud, split=split)

    # contiguous per-seq offsets (rows are pre-sorted by (seq_id, ts))
    n_seq = int(seq_id.max()) + 1
    starts = np.searchsorted(seq_id, np.arange(n_seq), side="left")
    ends = np.searchsorted(seq_id, np.arange(n_seq), side="right")
    np.savez(out / "seq_index.npz", starts=starts.astype(np.int64), ends=ends.astype(np.int64))

    print(f"[encode] codes {codes.shape} dtype={codes.dtype}; {n_seq:,} sequences")
    print(f"[encode] done in {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/transactions.parquet")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    tc = TokenizerConfig()
    ap.add_argument("--amount-buckets", type=int, default=tc.n_amount_buckets)
    ap.add_argument("--hash-buckets", type=int, default=tc.hash_buckets)
    ap.add_argument("--include-dt", action="store_true",
                    help="add a log-bucketed time-since-last-event field")
    ap.add_argument("--n-dt-buckets", type=int, default=tc.n_dt_buckets)
    ap.add_argument("--mname-cat", action="store_true",
                    help="(experiment a) encode merchant_name as a full identity vocab, not hashed")
    ap.add_argument("--zip-prefix", type=int, default=None,
                    help="(experiment b) replace zip with its N-digit regional prefix, as a cat")
    args = ap.parse_args()
    build(args.parquet, args.out_dir, args.tokenizer, args.amount_buckets, args.hash_buckets,
          args.include_dt, args.n_dt_buckets, tc.dt_min_s, tc.dt_max_s,
          mname_cat=args.mname_cat, zip_prefix=args.zip_prefix)


if __name__ == "__main__":
    main()
