"""Parse a time-contiguous TalkingData AdTracking subset into arrays for the relational gate.

TalkingData click-fraud is the real analogue of our synthetic mechanism: fraudulent IPs fire
click BURSTS across many apps in short windows, and the label (`is_attributed` = click led to an
app install) is rare (~0.2%). The features ship RAW (ip, app, device, os, channel, click_time) —
no pre-engineered neighbour aggregates — so we control the summary's richness, unlike Elliptic.

Mapping onto the thesis:
  local (blind)   = the click's own fields {app, device, os, channel, hour}   (IP identity excluded)
  + agg summary   = engineered IP-velocity features (as-of, causal)            (rank-1 analog)
  + raw neighbours= the IP's last-K prior clicks' raw fields                    (cross-sequence encoder)

The entity is the IP (the fraud-farm unit); IP identity is NEVER a feature (it is only the grouping
key), so the relational signal must come through aggregates or neighbours.

Writes {out}/td.npz: local (N,5) int32 codes [app,device,os,channel,hour], vocab sizes; agg (N,A)
f32 as-of IP-velocity features; y (N,) int8; ts (N,) int64; split (N,) int8 {0 train,1 test}. With
--with-neighbours also: nbr (N,K,4) int32 [app,device,os,channel] of last-K prior same-IP clicks,
nbr_dt (N,K) f32 seconds-to-target, nbr_mask (N,K) bool.

Run: python scripts/build_talkingdata.py --csv <subset.csv> --out data/td --with-neighbours --k 8
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="time-contiguous subset CSV (TalkingData train schema)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nrows", type=int, default=None, help="cap rows (already-contiguous input)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--with-neighbours", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.2, help="last fraction of time -> test")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.csv, nrows=args.nrows,
                     usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"])
    df["ts"] = pd.to_datetime(df["click_time"]).astype("int64") // 10**9        # epoch seconds
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    N = len(df)
    ts = df["ts"].to_numpy(np.int64)
    hour = ((ts // 3600) % 24).astype(np.int64)
    y = df["is_attributed"].to_numpy(np.int8)

    # local categorical codes (factorised; IP excluded from features)
    cats = {}
    local_cols = []
    for c in ["app", "device", "os", "channel"]:
        codes, _ = pd.factorize(df[c]); cats[c] = codes.astype(np.int32); local_cols.append(cats[c])
    local_cols.append(hour.astype(np.int32))
    local = np.column_stack(local_cols).astype(np.int32)                       # (N,5)
    vocab = [int(local[:, j].max()) + 1 for j in range(local.shape[1])]
    ip = pd.factorize(df["ip"])[0].astype(np.int64)                            # grouping key only

    # as-of IP-velocity aggregates (causal): prior count, windowed velocity (1h/10min), gap
    o = pd.DataFrame({"gidx": np.arange(N), "ip": ip, "ts": ts})
    o = o.sort_values(["ip", "ts"], kind="stable")
    g = o.groupby("ip", sort=False)
    prior = g.cumcount().to_numpy(np.float64)
    prev_ts = g["ts"].shift(1).to_numpy()
    gap = np.where(np.isnan(prev_ts), 1e7, o["ts"].to_numpy() - prev_ts)
    tsv = o["ts"].to_numpy(np.int64)
    idxmap = o.groupby("ip", sort=False).indices
    vel = {W: np.zeros(N) for W in (3600, 600)}
    for _, pos in idxmap.items():
        t = tsv[pos]
        for W in vel:
            vel[W][pos] = np.arange(len(t)) - np.searchsorted(t, t - W, side="left")
    agg_sorted = np.column_stack([np.log1p(prior), np.log1p(np.clip(gap, 0, None)),
                                  np.log1p(vel[3600]), np.log1p(vel[600])]).astype(np.float32)
    agg = np.empty_like(agg_sorted); agg[o["gidx"].to_numpy()] = agg_sorted    # back to global order

    # temporal split: last test-frac of TIME -> test
    thr = np.quantile(ts, 1 - args.test_frac)
    split = np.where(ts <= thr, 0, 1).astype(np.int8)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    payload = dict(local=local, vocab=np.array(vocab), agg=agg, y=y, ts=ts, split=split)

    if args.with_neighbours:
        K = args.k
        order = np.lexsort((ts, ip))                       # sort by (ip, ts)
        ip_s, ts_s = ip[order], ts[order]
        feat_s = local[order][:, :4].astype(np.int32)      # [app,device,os,channel] in sorted order
        nbr_s = np.zeros((N, K, 4), np.int32)
        dt_s = np.zeros((N, K), np.float32)
        msk_s = np.zeros((N, K), bool)
        for off in range(1, K + 1):                        # slot K-off = the off-th previous same-ip click
            same = np.zeros(N, bool)
            same[off:] = (ip_s[off:] == ip_s[:-off]) & (ts_s[off:] > ts_s[:-off])  # strict as-of
            src = np.arange(N) - off
            sel = same
            nbr_s[sel, K - off] = feat_s[src[sel]]
            dt_s[sel, K - off] = (ts_s[sel] - ts_s[src[sel]]).astype(np.float32)
            msk_s[sel, K - off] = True
        inv = np.empty(N, np.int64); inv[order] = np.arange(N)   # unsort to global order
        payload.update(nbr=nbr_s[inv], nbr_dt=dt_s[inv], nbr_mask=msk_s[inv])

    np.savez(out / "td.npz", **payload)
    tr, te = split == 0, split == 1
    print(f"[td] N={N:,} vocab(app,dev,os,ch,hr)={vocab}  neighbours={'yes' if args.with_neighbours else 'no'}")
    print(f"[td] train {int(tr.sum()):,} (pos {int(y[tr].sum()):,}, base {y[tr].mean():.4f}) | "
          f"test {int(te.sum()):,} (pos {int(y[te].sum()):,}, base {y[te].mean():.4f})  in {time.time()-t0:.0f}s")
    if args.with_neighbours:
        print(f"[td] neighbour coverage {payload['nbr_mask'].any(1).mean():.3f}")


if __name__ == "__main__":
    main()
