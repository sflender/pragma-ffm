"""Decisive relational-decomposition gate for the TalkingData 3-transformer test.

The earlier gate's "+0.17 from IP velocity" conflated two signals a 3-transformer FFM would
route through DIFFERENT components:
  - user-own history (user = ip+device+os, the standard TalkingData user proxy) -> History encoder
  - IP cross-user activity (the IP's clicks from OTHER users)                    -> 3rd transformer

For a genuine 3-transformer win, the signal must live in the CROSS-USER part. We measure it:
  local                         : the click's own fields
  + user_agg                    : user-own velocity/count (as-of)         [per-sequence proxy]
  + user_agg + ipcross_agg      : IP cross-user velocity (= ip - user)     [relational / 3rd-transformer]

The make-or-break number is delta_ipcross = (local+user+ipcross) - (local+user). If it is clearly
> 0, the cross-user IP signal exists beyond per-user history -> the 3rd transformer has headroom and
the full FFM run is worth it. If ~0, the signal is per-user (2nd transformer) and TalkingData is not
a 3-transformer story.

Run: python scripts/talkingdata_reldecomp.py --csv <subset.csv> --nrows 6000000
"""
from __future__ import annotations

import argparse, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def windowed(o_sorted_idx_by_group, tsv, windows):
    """as-of windowed counts per group; returns dict W -> array aligned to o order."""
    out = {W: np.zeros(len(tsv), np.float32) for W in windows}
    for _, pos in o_sorted_idx_by_group.items():
        t = tsv[pos]
        for W in windows:
            out[W][pos] = np.arange(len(t)) - np.searchsorted(t, t - W, side="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int, default=6000000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.csv, nrows=args.nrows,
                     usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"])
    df["ts"] = pd.to_datetime(df["click_time"]).astype("int64") // 10**9
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    N = len(df); ts = df["ts"].to_numpy(np.int64); y = df["is_attributed"].to_numpy(np.int8)
    hour = ((ts // 3600) % 24).astype(np.int32)

    ip = pd.factorize(df["ip"])[0].astype(np.int64)
    user = pd.factorize(pd.Series(list(zip(df["ip"], df["device"], df["os"]))))[0].astype(np.int64)
    W = (3600, 600)

    def agg_for(key):                                    # as-of prior count + windowed velocity for a grouping key
        o = pd.DataFrame({"g": key, "ts": ts, "gidx": np.arange(N)}).sort_values(["g", "ts"], kind="stable")
        prior = o.groupby("g", sort=False).cumcount().to_numpy(np.float64)
        idxmap = o.groupby("g", sort=False).indices
        vel = windowed(idxmap, o["ts"].to_numpy(np.int64), W)
        # back to global order
        inv = np.empty(N, np.int64); inv[o["gidx"].to_numpy()] = np.arange(N)
        prior_g = prior[inv]; vel_g = {w: vel[w][inv] for w in W}
        return prior_g, vel_g

    up, uv = agg_for(user)                               # user-own
    ipp, ipv = agg_for(ip)                               # ip total
    # cross-user IP activity = ip total - user own (clamp >=0)
    ipx_p = np.clip(ipp - up, 0, None)
    ipx_v = {w: np.clip(ipv[w] - uv[w], 0, None) for w in W}

    local = np.column_stack([pd.factorize(df[c])[0] for c in ["app", "device", "os", "channel"]] + [hour]).astype(np.float32)
    user_agg = np.column_stack([np.log1p(up)] + [np.log1p(uv[w]) for w in W]).astype(np.float32)
    ipcross_agg = np.column_stack([np.log1p(ipx_p)] + [np.log1p(ipx_v[w]) for w in W]).astype(np.float32)

    thr = np.quantile(ts, 1 - args.test_frac); split = (ts > thr)
    tr, te = ~split, split
    print(f"[rd] N={N:,} users={user.max()+1:,} ips={ip.max()+1:,} | "
          f"train {int(tr.sum()):,} (pos {int(y[tr].sum()):,}) test {int(te.sum()):,} (pos {int(y[te].sum()):,}, base {y[te].mean():.4f})  {time.time()-t0:.0f}s")
    print(f"[rd] mean cross-user IP vel(1h): pos {ipx_v[3600][y==1].mean():.1f} | neg {ipx_v[3600][y==0].mean():.1f}")

    import lightgbm as lgb
    def fit(X, tag):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                               class_weight="balanced", verbose=-1).fit(X[tr], y[tr])
        s = m.predict_proba(X[te])[:, 1]
        pr, roc = average_precision_score(y[te], s), roc_auc_score(y[te], s)
        print(f"[rd] {tag:24s} PR {pr:.3f} ROC {roc:.3f}"); return pr

    a = fit(local, "local")
    b = fit(np.hstack([local, user_agg]), "local+user")
    c = fit(np.hstack([local, user_agg, ipcross_agg]), "local+user+ipcross")
    print(f"[rd] delta_user    = {b-a:+.3f}  (per-sequence / History-encoder signal)")
    print(f"[rd] delta_ipcross = {c-b:+.3f}  (CROSS-USER / 3rd-transformer headroom — make-or-break)")


if __name__ == "__main__":
    main()
