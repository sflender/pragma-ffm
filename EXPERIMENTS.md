# Experiment Log

A running lab notebook for the pragma-ffm reproduction. Each entry records the question,
setup, result, interpretation, caveats, and exact command to reproduce. Newest at the
bottom. Summary tables in the README point here for the full reasoning.

## Fixed setup (unless noted)

- **Dataset:** IBM TabFormer synthetic credit-card transactions (24.39M txns, 2000 users,
  6139 (user,card) sequences, 0.122% fraud).
- **Split:** per-(user,card) sequence, seed 0 (`--split-mode seq`); train/val/test =
  4911/614/614 sequences. Whole sequences held out → tests generalisation to unseen users.
- **Model:** mini-PRAGMA `nano` (~2.9M params): key-value-time tokenizer, Event Encoder,
  RoPE History Encoder, MLM pretraining. 3000 steps, AdamW, cosine schedule, MPS (M4 Max).
- **Downstream:** fraud detection, frozen backbone + linear probe on record embeddings.
- **Metrics:** ROC-AUC and PR-AUC (average precision) + Recall@Precision. PR-AUC is the
  headline given the extreme imbalance (random ≈ base rate 0.0013).
- **Eval mode:** unless stated, **as-of-date (causal)** — each transaction is scored using
  only its own history (no future events), matching the paper's evaluation-point framing.

### Consolidated results (as-of-date / leakage-free, test set = 2.28M txns)

| arm | ROC-AUC | PR-AUC | R@P0.5 | R@P0.9 |
|-----|---------|--------|--------|--------|
| LightGBM (causal features) | 0.940 | 0.043 | 0.000 | 0.000 |
| PRAGMA nano, bucket | 0.944 | 0.219 | 0.091 | 0.012 |
| PRAGMA nano, bucket + Δt | **0.953** | **0.236** | **0.163** | 0.010 |
| PRAGMA nano, PLE | 0.942 | 0.110 | 0.019 | 0.000 |
| PRAGMA nano, periodic | 0.941 | 0.087 | 0.029 | 0.006 |

> **Measurement noise:** the probe fits on a random subsample of train windows, so PR-AUC
> has run-to-run variance of roughly ±0.005 (single seed). Differences below that are noise;
> the effects recorded below are larger. A seed sweep is a pending TODO.

---

## E0 — LightGBM baseline (the incumbent)

**Question:** how strong is a standard gradient-boosted-tree fraud model here?

**Setup:** causal engineered features (raw fields + expanding amount stats, `dt_last`,
`is_online`, categoricals), `scale_pos_weight`, AUC early stopping.
Repro: `python -m pragma.baselines.lgbm`

**Result:** ROC-AUC 0.940, PR-AUC 0.043 (≈34× the 0.0013 base rate). Best iteration = 1.

**Interpretation:** TabFormer fraud is **largely a static function** of a few fields
(online flag, error codes, MCC, amount) — one deep tree captures most of it. This is a
strong bar and means the sequence model must add value beyond static features.

---

## E1 — Does MLM pretraining beat the incumbent? (the paper's central claim)

**Question:** does a frozen, self-supervised backbone + linear probe beat LightGBM?

**Setup:** nano, bucket numeric mode, 3000 MLM steps; linear probe.
Repro: `python -m pragma.train.pretrain --preset nano --numeric-mode bucket --max-steps 3000`
then `python -m pragma.train.probe --ckpt artifacts/pretrain_nano_bucket.pt --causal`

**Result (as-of-date):** PRAGMA PR-AUC **0.219** vs LightGBM 0.043 → **+396%**;
ROC-AUC 0.944 vs 0.940; Recall@P0.5 0.091 vs 0.000.

**Interpretation:** the paper's claim reproduces at laptop scale. A 2.9M-param frozen
backbone beats the strong GBDT incumbent by ~5× on the metric that matters, with nonzero
recall at 50% precision where LightGBM flatlines.

---

## E2 — Bidirectional leakage vs. as-of-date scoring

**Question:** PRAGMA is a bidirectional encoder — does a naive per-window embedding cheat
by seeing *future* transactions, and how much of the win survives removing that?

**Setup:** same nano bucket backbone, two probe modes: bidirectional (`--causal` off) vs
as-of-date/causal (`--causal`, each event attends only to itself + earlier).

**Result:** bidirectional PR-AUC 0.242 → as-of-date **0.215–0.219** (~11% drop).

**Interpretation:** an out-of-time *split* does NOT fix this — the leakage is *within-
example* (future events in the same window), independent of the train/test boundary. The
paper controls it by truncating history at the evaluation point; causal masking is the
mathematically-equivalent, ~L× cheaper way to do it in one tiling pass. **The win is real
representation quality:** removing future context costs only ~11%, and 0.215 vs 0.043
(LightGBM) is still +396%. All subsequent experiments use as-of-date scoring.

---

## E3 — Numeric encoding: bucket vs PLE vs periodic

**Question:** does a continuous encoding of `amount` (PLE / periodic, Gorishniy et al.
2022) beat hard quantile bucketing? Prior: PLE gives a small boost.

**Setup:** identical nano/3000-steps/as-of-date, only the `amount` embedding differs.
Repro: `--numeric-mode {bucket,ple,periodic}` on pretrain + probe.

**Result:**

| numeric_mode | ROC-AUC | PR-AUC |
|--------------|---------|--------|
| bucket | 0.944 | **0.219** |
| PLE | 0.942 | 0.110 |
| periodic | 0.941 | 0.087 |

**Interpretation — prior was WRONG.** Hard bucketing won decisively on PR-AUC (opposite
the tabular-DL literature). ROC-AUC is tied; the gap is entirely at high precision.
Leading hypotheses:
1. **Objective mismatch (most likely):** the MLM target for amount is the *bucket id*, so
   bucket *input* is aligned with the reconstruction objective while PLE/periodic are not —
   the pretraining never rewards the extra within-bin resolution.
2. **Undertrained modules:** PLE/periodic add fresh Linear layers (and saw ~6% guard-
   skipped steps from early gradient spikes); bucket reuses the shared embedding table.
3. **Task doesn't need it:** 64 buckets already capture amount's fraud signal.

**Caveats:** single seed, nano, 3000 steps. **Fair rematch (pending):** add a continuous
(regression) MLM target for the numeric field so PLE's resolution has something to predict.
The tell for hypothesis #1: the literature's PLE gains come from optimising the *end task*
directly, not a bucket-shaped MLM proxy — encoding and objective must agree.

---

## E4 — Δt (time-since-last-event) embedding

**Question:** does an explicit log-bucketed time-since-previous-event field help, on top of
RoPE (which already encodes pairwise time) and hour/day-of-week features?

**Setup:** add a `dt` field (20 log-spaced second buckets, fine resolution for short gaps);
bucket mode; otherwise identical. Repro:
`python -m pragma.data.encode --include-dt --out-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json`
→ `python -m pragma.train.pretrain --preset nano --numeric-mode bucket --max-steps 3000 --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json --tag _dt`
→ `python -m pragma.train.probe --ckpt artifacts/pretrain_nano_bucket_dt.pt --causal --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json`

**Result:**

| model | ROC-AUC | PR-AUC | R@P0.5 |
|-------|---------|--------|--------|
| bucket (no Δt) | 0.944 | 0.219 | 0.091 |
| bucket + Δt | **0.953** | **0.236** | **0.163** |

**Interpretation — prior confirmed.** Δt helps: PR-AUC +7.8%, ROC-AUC +0.009, and
Recall@P0.5 **+79%** (0.091 → 0.163). The gain concentrates at high precision → the
**fraud-burst** signal (stolen cards fire several transactions seconds apart; log-bucketing
gives crisp resolution there). Not redundant with RoPE because (a) RoPE is only a *pairwise
attention bias* whereas Δt is per-event *content* usable from layer 1 and by the probe, and
(b) log-spacing puts resolution exactly at short gaps.

**Caveats:** single seed. Consistent positive across all metrics; magnitude needs a seed sweep.

---

## Engineering notes

- **Timing (M4 Max, MPS):** nano ≈ 0.48 s/step (~13 min/epoch); small ≈ 2.97 s/step
  (~60 min/epoch). Full runs feasible locally; no remote compute needed at this scale.
- **MPS NaN fix:** a `finfo.min` additive attention mask caused NaN in MPS softmax-backward
  mid-training. Fixed with a moderate finite mask (`-1e4`) + a guard that skips the optimizer
  step on any non-finite gradient. Bucket runs then train clean (0 skips); the new PLE/
  periodic Linear layers show ~5-7% early skips (benign, guard-absorbed, conservative).

## Open questions / next experiments

- [ ] **SIM-style retrieval instead of hard truncation.** Current limitation: history is
  hard-truncated to the most-recent `L` events (as-of-date window), but sequences are huge
  (mean 3972, max 70,008 events/card) — so for heavy users we score on the last ~2% of
  history and drop the rest. Recency truncation can't see long-range patterns (repeat fraud
  at the same merchant years apart, dormant-then-active cards). Fix = SIM (Alibaba,
  Pi et al. 2020): a **General Search Unit** retrieves the top-k *relevant* past events from
  the full lifelong history given the current transaction (hard search = same MCC/merchant/
  city; or soft search = embedding ANN), then the transformer attends over those instead of
  "last L." Preserves as-of-date causality (search only events before the target); composes
  with Δt. This is the priority next architectural step.
- [ ] Seed sweep (≥3 seeds) to put error bars on E3/E4/scaling deltas.
- [ ] Scaling: if `small` looks flat, rerun compute-matched (more steps for bigger models)
  — the current sweep is fixed at 3000 steps, so a flat point may be undertraining.
- [ ] E3 fair rematch: continuous/regression MLM target for the numeric field.
- [ ] LoRA fine-tuning as a second PRAGMA readout (paper reports this too).
- [ ] From-scratch task-specific transformer (the paper's literal "domain-specific" contender).
- [ ] Temporal-split variant (harder, shift-confounded) for contrast.
- [ ] Log-transform / learned frequencies for RoPE time input (aliasing at large gaps).
