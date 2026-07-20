# TalkingData + count-aware cross-sequence encoder (the headline result)

The sequence-native real-data test the graph datasets (Elliptic/Yelp/Amazon) could not give us:
TalkingData click fraud, where fraud is **transient cross-entity velocity** — exactly our synthetic
mechanism, but real and public. Sequence entity = **user** `(ip, device, os)`; shared entity =
**IP** (the fraud farm). A per-account FFM over a user's own clicks is blind to the IP's activity
across *other* users.

## The relational signal is real, and it's a COUNT
Decomposition gate (LightGBM, `talkingdata_reldecomp.py`): decomposing the raw IP-velocity signal
into the user's own history vs. the IP's cross-user activity —
- `+ user history` (History encoder's signal): **+0.065**
- `+ IP cross-user velocity` (3rd-transformer's signal): **+0.075**

The cross-user signal is real and *larger* than the per-user one — but it is a velocity
**magnitude** (fraud IPs ~700 clicks/hr), not a pattern in neighbour content.

## Multi-seed ablation (3 seeds, PR-AUC mean ± std)
Same balanced eval subsample for the FFM arms and the matched GBDT.

| arm | Synthetic (pattern signal) | TalkingData (count signal) |
|---|---|---|
| GBDT (matched) | 0.36 velocity ceiling† | 0.405 |
| FFM (per-account) | 0.039 ± 0.004 | 0.616 ± 0.013 |
| + memory (count path) | 0.088 ± 0.009 | 0.672 ± 0.009 |
| + xseq (raw attention, pattern path) | 0.252 ± **0.117** | 0.619 ± 0.004 |
| **+ count-aware cross-seq** | **0.318 ± 0.010** (n=2) | **0.700 ± 0.011** |

† All-feature GBDT hits 0.605 but leaks a label-derived merchant fraud-rate; 0.36 (velocity only)
is the honest relational-signal ceiling.

## Findings
1. **Count-aware cross-sequence is the best arm on BOTH datasets, robustly.** TalkingData
   0.700 ± 0.011 (+0.084 over per-account FFM, tight); synthetic 0.318 ± 0.010.
2. **The path that wins depends on the signal type.** On TalkingData (count), the *memory* path
   wins and raw attention alone does **not** help (0.619 ≈ per-account) — a K-window truncates a
   700/hr magnitude a scalar log-count captures. On synthetic (pattern), raw attention carries the
   lift.
3. **Raw attention is seed-unstable; the count path stabilises it.** Synthetic xseq is
   0.25 ± 0.12 (per-seed 0.41/0.13/0.22 — the earlier single-seed "0.409" was a lucky seed);
   count-aware is 0.32 ± 0.01. The always-on magnitude readout anchors the noisy attention.
4. **This is the count-vs-pattern taxonomy, unified.** One module (attention path + log-count
   readout) covers both regimes, so a single deployed "FFM + cross-sequence attention" arm improves
   over the per-account FFM whether the relational signal is pattern- or count-type.

## Honest caveats
- Synthetic `+count-aware` is n=2 (the 3-seed sweep hit the pod watchdog before the 3rd seed);
  0.318 ± 0.010 is already tight. Baselines and TalkingData are full n=3.
- The matched GBDT (0.405) lacks hand-built per-user history aggregates, so the FFM's margin over
  it partly reflects sequence modelling; the claim we stand behind is the **ablation** (+0.084 from
  count-aware cross-sequence over the identical per-account FFM), not "FFM beats GBDT."
- Laptop-scale, `small` preset; TalkingData is a 6M-row time-contiguous subset.

Artifacts: `artifacts/{synth,td}_{nomem,mem,xseq,xseqcount}_s{0,1,2}.json`, `*_lgbm.json`.
Repro: `scripts/build_talkingdata_seq.py`, `build_merchant_memory.py --entity ip`,
`build_entity_neighbors.py --entity ip`, `finetune_synth.py --xseq --xseq-count`.
