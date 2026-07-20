"""Controlled synthetic transaction generator with *tunable* fraud structure.

The point: create data where we KNOW where the fraud signal lives, so we can test the
central thesis cleanly — a per-sequence FFM should capture fraud that is a function of a
card's OWN history, and should FAIL on fraud that is a function of a shared entity's
cross-card state (which a per-sequence model structurally cannot see). A relational model
(entity memory / cross-sequence attention) should recover the latter.

Two fraud mechanisms (`--mode`):
  per_card    : a card is compromised at a random time -> a short BURST of fraud txns on
                that card. Detectable from the card's own recent velocity (per-sequence).
  relational  : a *merchant* is compromised for a window -> a velocity spike of fraud txns
                at that merchant, spread across MANY random cards. Fraud amounts are drawn
                from the SAME distribution as legit, so there is NO per-transaction tell —
                the only signal is the merchant's transient cross-card velocity / recent
                fraud. Invisible to a per-sequence model; recoverable from merchant memory.
  mix         : fraction `--relational-frac` of fraud is relational, the rest per_card.

Output: the canonical parquet the pipeline eats (seq_id, ts, event_idx, amount, is_fraud,
split, card, merchant, mcc). Encode with `--dataset synth`.

Run: python scripts/gen_synth_relational.py --mode relational --out data/synth_rel/transactions.parquet
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

DAY = 86400
REF = 1_600_000_000


def _base_transactions(rng, n_cards, n_merch, span_days, mean_len):
    """Legit background: each card fires ~Poisson(mean_len) txns over the span at Zipf merchants."""
    n_per = rng.poisson(mean_len, n_cards).clip(2)
    N = int(n_per.sum())
    card = np.repeat(np.arange(n_cards), n_per)
    ts = (rng.uniform(0, span_days * DAY, N)).astype(np.int64)
    # merchant popularity ~ Zipf; mcc = merchant's category (deterministic)
    zipf = 1.0 / np.arange(1, n_merch + 1)
    merch = rng.choice(n_merch, size=N, p=zipf / zipf.sum())
    mcc = (merch % 20)
    amount = np.exp(rng.normal(3.2, 1.0, N)).astype(np.float32)          # lognormal spend
    df = pd.DataFrame({"card": card, "merchant": merch, "mcc": mcc, "ts": ts,
                       "amount": amount, "is_fraud": np.zeros(N, np.int8)})
    return df


def _inject_per_card(rng, df, target_rate):
    """Compromise a subset of cards -> a burst of fraud txns appended to each at a random time."""
    cards = df["card"].unique()
    n_comp = int(len(cards) * target_rate * 6)                            # tune to hit target rate
    victims = rng.choice(cards, size=min(n_comp, len(cards)), replace=False)
    rows = []
    for c in victims:
        t0 = rng.uniform(0, 175 * DAY)
        k = rng.integers(3, 9)                                            # burst length
        for j in range(k):
            m = rng.integers(0, df["merchant"].max() + 1)
            rows.append((c, m, m % 20, int(t0 + j * rng.integers(30, 600)),
                         float(np.exp(rng.normal(3.2, 1.0))), 1))
    fr = pd.DataFrame(rows, columns=["card", "merchant", "mcc", "ts", "amount", "is_fraud"])
    fr["is_fraud"] = fr["is_fraud"].astype(np.int8)
    return pd.concat([df, fr], ignore_index=True)


def _inject_relational(rng, df, target_rate):
    """Compromise merchants in short windows -> velocity spikes of fraud across many cards.
    Fraud is allocated to EVERY merchant ∝ its legit traffic (uniform per-merchant fraud rate
    => merchant/mcc IDENTITY is uninformative), and clustered into temporal BURSTS (=> the only
    tell is the merchant's transient cross-card velocity). Fraud amounts ~ legit (no per-txn
    tell); fraud cards are random (=> no per-CARD-history tell either). A per-sequence model
    is structurally blind to this; only merchant memory / cross-sequence attention sees it."""
    cards = df["card"].unique()
    rate_factor = target_rate / (1 - target_rate)
    rows = []
    for m, sub in df.groupby("merchant"):
        f_m = int(round(rate_factor * len(sub)))
        if f_m < 2:
            continue
        tmin, tmax = int(sub["ts"].min()), int(sub["ts"].max())
        n_bursts = max(1, f_m // 12)                                     # ~12 fraud txns per burst
        per = max(1, f_m // n_bursts)
        for ct in rng.uniform(tmin, tmax, n_bursts):
            W = rng.integers(2 * 3600, 8 * 3600)                        # short compromise window
            for _ in range(per):
                rows.append((int(rng.choice(cards)), int(m), int(m) % 20,
                             int(ct + rng.uniform(0, W)),
                             float(np.exp(rng.normal(3.2, 1.0))), 1))
    fr = pd.DataFrame(rows, columns=["card", "merchant", "mcc", "ts", "amount", "is_fraud"])
    fr["is_fraud"] = fr["is_fraud"].astype(np.int8)
    return pd.concat([df, fr], ignore_index=True)


def _inject_pattern(rng, df, target_rate):
    """Compromised-merchant bursts whose fraud txns carry a CONTENT SIGNATURE, not only a velocity
    spike. Each merchant normally transacts at a single MCC (= merchant % 20). During a compromise
    window the fraud txns (spread across MANY random cards) use a DIFFERENT but globally-valid MCC
    (usual+7 mod 20) and a distinctive amount band. So:
      - a per-transaction tell is weak: the signature MCC is a normal MCC for *other* merchants, and
        the amount overlaps the legit range, so no single txn is a giveaway;
      - the tell is RELATIONAL + CONTENT: it only shows up as an inconsistency with the merchant's
        recent neighbours (which all sit at the merchant's usual MCC) -- readable by cross-sequence
        attention over neighbour content, but not by a scalar velocity/mean aggregate.
    Random fraud cards => a per-sequence model stays blind, exactly as in the count mode."""
    cards = df["card"].unique()
    rate_factor = target_rate / (1 - target_rate)
    rows = []
    for m, sub in df.groupby("merchant"):
        f_m = int(round(rate_factor * len(sub)))
        if f_m < 2:
            continue
        sig_mcc = (int(m) + 7) % 20                                    # valid MCC, != this merchant's usual
        tmin, tmax = int(sub["ts"].min()), int(sub["ts"].max())
        for _ in range(f_m):
            # SPREAD over the merchant's timeline (no burst) => velocity is NOT a tell; and amounts
            # drawn from the legit distribution => no per-txn / mean-amount tell. The ONLY signal is
            # that this txn's MCC is inconsistent with the merchant's usual MCC -- visible only by
            # reading the merchant's recent neighbours' content, not from any scalar aggregate.
            mcc_i = sig_mcc if rng.random() < 0.8 else (int(m) % 20)  # 20% "stealth": no content tell
            rows.append((int(rng.choice(cards)), int(m), mcc_i,
                         int(rng.uniform(tmin, tmax)),
                         float(np.exp(rng.normal(3.2, 1.0))), 1))
    fr = pd.DataFrame(rows, columns=["card", "merchant", "mcc", "ts", "amount", "is_fraud"])
    fr["is_fraud"] = fr["is_fraud"].astype(np.int8)
    return pd.concat([df, fr], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth_rel/transactions.parquet")
    ap.add_argument("--mode", choices=["per_card", "relational", "pattern", "mix"], default="relational")
    ap.add_argument("--relational-frac", type=float, default=0.5, help="mix: fraction relational")
    ap.add_argument("--n-cards", type=int, default=4000)
    ap.add_argument("--n-merchants", type=int, default=400)
    ap.add_argument("--span-days", type=int, default=180)
    ap.add_argument("--mean-len", type=int, default=40)
    ap.add_argument("--target-fraud-rate", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    df = _base_transactions(rng, args.n_cards, args.n_merchants, args.span_days, args.mean_len)
    if args.mode == "per_card":
        df = _inject_per_card(rng, df, args.target_fraud_rate)
    elif args.mode == "relational":
        df = _inject_relational(rng, df, args.target_fraud_rate)
    elif args.mode == "pattern":
        df = _inject_pattern(rng, df, args.target_fraud_rate)
    else:
        df = _inject_per_card(rng, df, args.target_fraud_rate * (1 - args.relational_frac))
        df = _inject_relational(rng, df, args.target_fraud_rate * args.relational_frac)

    # canonical formatting
    df["ts"] = (REF + df["ts"]).astype(np.int64)
    df = df.sort_values(["card", "ts"], kind="stable").reset_index(drop=True)
    df["seq_id"] = df["card"].astype("category").cat.codes.astype(np.int32)
    df = df.sort_values(["seq_id", "ts"], kind="stable").reset_index(drop=True)
    df["event_idx"] = df.groupby("seq_id").cumcount().astype(np.int32)
    df["merchant"] = df["merchant"].astype(str)
    df["mcc"] = df["mcc"].astype(str)

    # per-card split (unseen cards), seed 0
    uniq = np.unique(df["seq_id"].to_numpy()); rng2 = np.random.default_rng(0)
    perm = rng2.permutation(uniq.size); n_tr, n_va = int(.8 * uniq.size), int(.9 * uniq.size)
    code = np.empty(uniq.size, dtype=object)
    code[perm[:n_tr]] = "train"; code[perm[n_tr:n_va]] = "val"; code[perm[n_va:]] = "test"
    df["split"] = df["seq_id"].map(dict(zip(uniq.tolist(), code.tolist())))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = df[["seq_id", "ts", "event_idx", "amount", "is_fraud", "split", "card", "merchant", "mcc"]]
    df.to_parquet(args.out, index=False)
    rep = df.groupby("split").agg(n=("is_fraud", "size"), f=("is_fraud", "sum"))
    print(f"[synth] mode={args.mode} wrote {args.out}  {len(df):,} rows  "
          f"fraud {df['is_fraud'].mean():.4f}  seqs {df['seq_id'].nunique():,}  in {time.time()-t0:.0f}s")
    print(rep.to_string())
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(
        {"dataset": "synth", "mode": args.mode, "n_rows": int(len(df)),
         "fraud_rate": float(df["is_fraud"].mean())}, indent=2))


if __name__ == "__main__":
    main()
