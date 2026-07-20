"""TalkingData -> canonical parquet for the FULL 3-transformer FFM (not the tabular gate).

SEQUENCE entity = user = (ip, device, os) (the standard TalkingData user proxy); the shared
cross-user entity is the IP. The per-sequence FFM (event encoder over each click's fields +
history encoder over the user's own click stream) is structurally blind to the IP's activity
across OTHER users; the cross-IP memory / cross-sequence encoder (3rd transformer) sees it.

Emits the canonical schema the generic pipeline eats (encode with --dataset talkingdata):
  seq_id (=user), ts, event_idx, is_fraud (=is_attributed), split, amount (dummy 0),
  app, channel, device, os, ip  (ip kept for the cross-IP memory/neighbour builders).

Run: python scripts/build_talkingdata_seq.py --csv <subset.csv> --out data/td_seq/transactions.parquet
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/td_seq/transactions.parquet")
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.csv, nrows=args.nrows,
                     usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"])
    df["ts"] = pd.to_datetime(df["click_time"]).astype("int64") // 10**9
    # user = (ip, device, os); seq_id is its dense code
    user = pd.factorize(pd.Series(list(zip(df["ip"], df["device"], df["os"]))))[0]
    df["seq_id"] = user.astype(np.int32)
    df["ip"] = pd.factorize(df["ip"])[0].astype(np.int32)
    df["is_fraud"] = df["is_attributed"].astype(np.int8)
    df["amount"] = np.float32(0.0)                        # clicks have no amount; dummy for the encoder

    # sort into contiguous per-seq, time-ordered rows
    df = df.sort_values(["seq_id", "ts"], kind="stable").reset_index(drop=True)
    df["event_idx"] = df.groupby("seq_id").cumcount().astype(np.int32)

    # temporal split by global click time (last 20% -> test, prior 5% -> val)
    ts = df["ts"].to_numpy(np.int64)
    q75, q80 = np.quantile(ts, [0.75, 0.80])
    split = np.where(ts > q80, "test", np.where(ts > q75, "val", "train"))
    df["split"] = split

    cols = ["seq_id", "ts", "event_idx", "is_fraud", "split", "amount",
            "app", "channel", "device", "os", "ip"]
    df = df[cols]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    rep = df.groupby("split").agg(n=("is_fraud", "size"), f=("is_fraud", "sum"))
    lens = df.groupby("seq_id").size()
    print(f"[td-seq] wrote {args.out}  {len(df):,} rows  users(seq)={df['seq_id'].nunique():,} "
          f"ips={df['ip'].nunique():,}  fraud {df['is_fraud'].mean():.4f}  "
          f"seqlen mean {lens.mean():.1f} median {lens.median():.0f}  in {time.time()-t0:.0f}s")
    print(rep.to_string())
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(
        {"dataset": "talkingdata", "n_rows": int(len(df)), "fraud_rate": float(df["is_fraud"].mean())}, indent=2))


if __name__ == "__main__":
    main()
