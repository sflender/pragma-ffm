"""Build YelpChi / Amazon review-fraud graphs into the Elliptic-format arrays, so the same
node-arms (`scripts/elliptic_relational.py`) run unchanged.

These are the content/homophily regime our other real datasets lacked: fraudsters connect to
fraudsters, so a neighbour's *features* (not just a velocity count) are discriminative — the case
where raw-neighbour attention should beat a summary, and where graph methods are documented to beat
MLP (CARE-GNN, PC-GNN). No pre-shipped neighbour aggregates, so we control the summary:

  x_local = the node's own features (blind / per-node)
  x_agg   = MEAN of ALL neighbours' features (GraphSAGE-mean summary — the "+agg" arm)
  nbr     = K sampled neighbours' raw features (the "+xseq" attention arm)

Uses the combined `homo` relation as the graph. Stratified random split (no time axis).

Run: python scripts/build_yelpamazon.py --mat <YelpChi.mat|Amazon.mat> --out data/yelp --k 16
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    m = sio.loadmat(args.mat)
    feat = m["features"]; feat = feat.toarray() if sp.issparse(feat) else np.asarray(feat)
    feat = feat.astype(np.float32)
    y = np.asarray(m["label"]).squeeze().astype(np.int8)
    A = m["homo"].tocsr() if sp.issparse(m["homo"]) else sp.csr_matrix(m["homo"])
    A.setdiag(0); A.eliminate_zeros()                     # drop self-loops
    N, F = feat.shape
    rng = np.random.default_rng(args.seed)

    # stratified split
    split = np.full(N, -1, np.int8)
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        ntr = int(len(idx) * args.train_frac)
        split[idx[:ntr]] = 0; split[idx[ntr:]] = 1
    tr = split == 0

    # z-score features on train
    mu, sd = feat[tr].mean(0), feat[tr].std(0); sd[sd < 1e-6] = 1.0
    xl = ((feat - mu) / sd).astype(np.float32)

    # x_agg = mean of ALL neighbours' (z-scored) features  -> GraphSAGE-mean summary
    deg = np.asarray(A.sum(1)).squeeze(); deg[deg == 0] = 1
    Dinv = sp.diags(1.0 / deg)
    xa = (Dinv @ A @ xl).astype(np.float32)

    # K sampled neighbours per node -> nbr_idx / mask
    K = args.k
    nbr_idx = np.full((N, K), -1, np.int32); nbr_mask = np.zeros((N, K), bool)
    indptr, indices = A.indptr, A.indices
    for i in range(N):
        nb = indices[indptr[i]:indptr[i + 1]]
        if len(nb) == 0:
            continue
        if len(nb) > K:
            nb = rng.choice(nb, K, replace=False)
        nbr_idx[i, :len(nb)] = nb; nbr_mask[i, :len(nb)] = True

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "graph.npz", x_local=xl, x_agg=xa, y=y, step=np.zeros(N, np.int16),
             split=split, nbr_idx=nbr_idx, nbr_mask=nbr_mask)
    te = split == 1
    print(f"[ya] {Path(args.mat).stem}: N={N:,} F={F} fraud={y.mean():.3f}  "
          f"deg mean {deg.mean():.0f} median {np.median(deg):.0f}  cover {(nbr_mask.any(1)).mean():.3f}")
    print(f"[ya] train {int(tr.sum()):,} (fraud {int(y[tr].sum()):,}) | "
          f"test {int(te.sum()):,} (fraud {int(y[te].sum()):,}, base {y[te].mean():.3f})  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
