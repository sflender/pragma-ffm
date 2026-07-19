"""Parse the Elliptic Bitcoin dataset into arrays for the relational-fraud external-validity test.

Elliptic is a transaction GRAPH (nodes = txns, edges = BTC flow), the real analogue of our
synthetic cross-entity signal. Its 165 features split into LOCAL (the node's own tx features) and
AGGREGATED (hand-engineered one-hop neighbour summaries) — mapping onto our thesis:
  local-only            = per-sequence / per-node model (blind to relations)
  local + aggregated    = rank-1 precomputed neighbour summary (like our merchant memory)
  local + raw-neighbour = the cross-sequence encoder (attend over neighbours' RAW features)

Standard convention: col0=txId, col1=time_step(1..49), cols 2..94 = 93 local, cols 95..166 = 72
aggregated. Standard temporal split: steps 1..34 train, 35..49 test (Weber et al. 2019). Labels:
class 1 = illicit (positive), 2 = licit (negative), unknown = unlabeled (kept as neighbours only).

Writes {out}/elliptic.npz:
  x_local (N,93) f32, x_agg (N,72) f32, y (N,) int8 {1 illicit,0 licit,-1 unknown},
  step (N,) int8, split (N,) int8 {0 train,1 test,-1 unlabeled/other},
  nbr_idx (N,K) int32 row-indices of up-to-K undirected neighbours (-1 pad), nbr_mask (N,K) bool.

Run: python scripts/build_elliptic.py --raw <dir with the 3 csvs> --out data/elliptic --k 16
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing elliptic_txs_{features,classes,edgelist}.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16, help="max neighbours per node")
    ap.add_argument("--train-max-step", type=int, default=34, help="steps <= this -> train, else test")
    args = ap.parse_args()

    t0 = time.time()
    raw = Path(args.raw)
    feat = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    cls = pd.read_csv(raw / "elliptic_txs_classes.csv")
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    N = len(feat)

    txid = feat.iloc[:, 0].to_numpy(np.int64)
    step = feat.iloc[:, 1].to_numpy(np.int64)
    x_local = feat.iloc[:, 2:95].to_numpy(np.float32)      # 93 local features
    x_agg = feat.iloc[:, 95:167].to_numpy(np.float32)      # 72 aggregated (neighbour) features
    assert x_local.shape[1] == 93 and x_agg.shape[1] == 72, (x_local.shape, x_agg.shape)

    # labels aligned to feature rows
    row = {int(t): i for i, t in enumerate(txid)}
    y = np.full(N, -1, np.int8)
    cmap = {"1": 1, "2": 0}                                # illicit->1, licit->0, unknown->-1
    for t, c in zip(cls["txId"].to_numpy(np.int64), cls["class"].astype(str).to_numpy()):
        if c in cmap and int(t) in row:
            y[row[int(t)]] = cmap[c]

    # temporal split (labeled only); unlabeled and other -> -1
    split = np.full(N, -1, np.int8)
    lab = y >= 0
    split[lab & (step <= args.train_max_step)] = 0
    split[lab & (step > args.train_max_step)] = 1

    # undirected adjacency (in+out), row-indexed
    e1 = np.array([row[t] for t in edges["txId1"].to_numpy(np.int64)], np.int32)
    e2 = np.array([row[t] for t in edges["txId2"].to_numpy(np.int64)], np.int32)
    adj = [[] for _ in range(N)]
    for a, b in zip(e1, e2):
        adj[a].append(b); adj[b].append(a)

    K = args.k
    nbr_idx = np.full((N, K), -1, np.int32)
    nbr_mask = np.zeros((N, K), bool)
    deg = np.zeros(N, np.int32)
    rng = np.random.default_rng(0)
    for i in range(N):
        nb = adj[i]
        deg[i] = len(nb)
        if not nb:
            continue
        nb = np.array(nb, np.int32)
        if len(nb) > K:                                    # sample K (seeded) for high-degree hubs
            nb = rng.choice(nb, size=K, replace=False)
        nbr_idx[i, :len(nb)] = nb
        nbr_mask[i, :len(nb)] = True

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "elliptic.npz", x_local=x_local, x_agg=x_agg, y=y, step=step, split=split,
             nbr_idx=nbr_idx, nbr_mask=nbr_mask)
    tr, te = split == 0, split == 1
    print(f"[elliptic] N={N:,}  local={x_local.shape[1]} agg={x_agg.shape[1]}  "
          f"labeled={int(lab.sum()):,} (illicit {int((y==1).sum()):,})  in {time.time()-t0:.0f}s")
    print(f"[elliptic] train {int(tr.sum()):,} (illicit {int((y[tr]==1).sum()):,}, "
          f"base {y[tr].clip(0,1).mean():.3f}) | test {int(te.sum()):,} "
          f"(illicit {int((y[te]==1).sum()):,}, base {y[te].clip(0,1).mean():.3f})")
    print(f"[elliptic] degree: mean {deg.mean():.1f} median {np.median(deg):.0f} "
          f"p95 {np.percentile(deg,95):.0f} max {deg.max()}  cover(>=1 nbr) {(deg>0).mean():.3f}")


if __name__ == "__main__":
    main()
