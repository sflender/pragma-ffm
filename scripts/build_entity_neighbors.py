"""Precompute each event's last-K prior ENTITY neighbours for the cross-sequence encoder.

The rank-1 ``merchant_mem`` compresses a merchant's recent cross-card activity to a handful
of hand-designed features (velocity, prior fraud rate...). The "third transformer" instead
consumes the RAW neighbour events and learns what to extract. This script builds, for every
transaction g, the last K events at the same entity (e.g. merchant) with ``ts`` strictly
before g (causal / leakage-safe, across ALL cards) — the exact window in which a compromised
merchant's velocity burst shows up.

Aligned to the global (seq_id, ts) row order of ``encoded.npz`` (same alignment as
build_merchant_memory.py), written to ``{data-dir}/entity_nbr.npz``:
  codes  (N, K, F) int16  neighbour field codes (front-padded)     -- reuses encoded.npz codes
  amount (N, K)   float32 neighbour raw amounts
  dt     (N, K)   float32 seconds from neighbour to target (>=0)
  mask   (N, K)   bool    True = real neighbour

Run:  python scripts/build_entity_neighbors.py --parquet data/synth_rel/transactions.parquet \
          --data-dir data/synth_rel --entity merchant --k 16
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--entity", default="merchant", help="entity column to gather neighbours over")
    ap.add_argument("--k", type=int, default=16, help="#prior same-entity neighbours per target")
    args = ap.parse_args()

    t0 = time.time()
    enc = np.load(Path(args.data_dir) / "encoded.npz")
    codes = enc["codes"]                                   # (N, F) int16, aligned to parquet rows
    amount = enc["amount_raw"].astype(np.float32)          # (N,)
    ts = enc["ts"].astype(np.int64)                        # (N,)
    N, F, K = codes.shape[0], codes.shape[1], args.k

    df = pd.read_parquet(args.parquet, columns=[args.entity])
    assert len(df) == N, f"parquet rows {len(df)} != encoded rows {N} (alignment broken)"
    merch = pd.factorize(df[args.entity])[0].astype(np.int64)   # (N,)

    nbr_codes = np.zeros((N, K, F), dtype=np.int16)
    nbr_amount = np.zeros((N, K), dtype=np.float32)
    nbr_dt = np.zeros((N, K), dtype=np.float32)
    nbr_mask = np.zeros((N, K), dtype=bool)

    order = np.lexsort((ts, merch))                        # sort by (merch, ts)
    m_sorted = merch[order]
    grp_starts = np.searchsorted(m_sorted, np.unique(m_sorted))
    grp_bounds = np.append(grp_starts, N)

    for gi in range(len(grp_starts)):                     # per entity group (ts-ascending)
        pos = order[grp_bounds[gi]:grp_bounds[gi + 1]]    # global rows, ts-ascending
        g_ts = ts[pos]
        for j in range(len(pos)):                         # target = pos[j]
            lo = max(0, j - K)
            cand = pos[lo:j]                              # up-to-K immediately-prior at entity
            if cand.size == 0:
                continue
            valid = g_ts[lo:j] < g_ts[j]                  # strict as-of (drop same-ts ties)
            cand = cand[valid]
            if cand.size == 0:
                continue
            r = int(pos[j]); c = cand.size
            nbr_codes[r, K - c:] = codes[cand]            # front-pad, most-recent last
            nbr_amount[r, K - c:] = amount[cand]
            nbr_dt[r, K - c:] = (g_ts[j] - ts[cand]).astype(np.float32)
            nbr_mask[r, K - c:] = True

    out = Path(args.data_dir) / "entity_nbr.npz"
    np.savez(out, codes=nbr_codes, amount=nbr_amount, dt=nbr_dt, mask=nbr_mask)
    cov = nbr_mask.any(1).mean()
    print(f"[nbr] wrote {out}  codes={nbr_codes.shape} K={K}  "
          f"coverage={cov:.3f} (rows with >=1 neighbour)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
