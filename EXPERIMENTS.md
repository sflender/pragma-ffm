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

> ⚠️ **The tables in E0–E9 use eval method (b)** (causal-masked tiling), later found to
> under-context events and deviate from PRAGMA. **See [E10](#e10) for the corrected
> method-(a) numbers** (sliding-window as-of-date) — the ones to cite. Qualitative findings
> mostly hold; the Δt effect shrinks from +0.05 to +0.006 under (a).

| arm | ROC-AUC | PR-AUC | R@P0.5 | R@P0.9 |
|-----|---------|--------|--------|--------|
| LightGBM (causal features) | 0.940 | 0.043 | 0.000 | 0.000 |
| PRAGMA nano, bucket | 0.944 | 0.219 | 0.091 | 0.012 |
| PRAGMA nano, bucket + Δt | 0.953 | 0.236 | 0.163 | 0.010 |
| PRAGMA mini, bucket + Δt (7.5M) | 0.959 | 0.364 | 0.358 | 0.086 |
| PRAGMA small, bucket + Δt (13.7M, 3k steps) | 0.963 | 0.349 | 0.310 | 0.102 |
| **PRAGMA small, bucket + Δt (13.7M, 6k steps)** | **0.977** | **0.495** | **0.523** | **0.193** |
| PRAGMA nano, PLE | 0.942 | 0.110 | 0.019 | 0.000 |
| PRAGMA nano, periodic | 0.941 | 0.087 | 0.029 | 0.006 |

> **Measurement noise:** the probe fits on a random subsample of train windows, so PR-AUC
> has run-to-run variance of roughly ±0.005 (now seeded/deterministic). **Training-seed
> variance is far larger: nano@3000 swings ±~0.05 PR-AUC across seeds (E9)** — as big as many
> single-seed deltas here. Trust effects that clear ±0.05 (E3, E5/E6 scaling); treat smaller
> single-seed nano deltas (E4) as suggestive. Multi-seed everything for firm claims.

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
**UPDATE (E9):** a 3-seed run of this nano+Δt config gives mean PR-AUC **0.164 ± 0.057** —
seed 0 (0.236) was a high draw. The +Δt gain (+0.017) is *within* nano seed noise (±0.05), so
treat E4 as **suggestive, not established**. Needs a proper seed sweep (or a larger, more
stable model) to confirm.

---

## E5 — Model scaling (nano → mini → small)

**Question:** does the FFM keep improving with capacity, or is this task saturated?

**Setup:** three sizes, all bucket + Δt, **fixed 3000 steps**, as-of-date probe.
Repro: `python -m pragma.train.pretrain --preset {nano,mini,small} --numeric-mode bucket --max-steps 3000 --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json --tag _dt` then probe `--causal`.

**Result:**

| size | params | ROC-AUC | PR-AUC | R@P0.5 | R@P0.9 | final MLM loss |
|------|--------|---------|--------|--------|--------|----------------|
| nano | 2.9M | 0.9525 | 0.2358 | 0.163 | 0.010 | 2.05 |
| mini | 7.5M | 0.9586 | **0.3644** | **0.358** | 0.086 | 1.69 |
| small | 13.7M | **0.9632** | 0.3489 | 0.310 | **0.102** | **1.57** |

**Interpretation — splits by metric:**
- **ROC-AUC and pretraining loss scale monotonically** (loss 2.05 → 1.69 → 1.57): as a
  language model, bigger is strictly better, no saturation.
- **PR-AUC and Recall@P0.5 peak at `mini` and dip at `small`** (0.364 → 0.349; 0.358 →
  0.310). Dips exceed probe noise (~±0.005), so it's a real bend.
- Overturns the earlier "static/saturated task" read: nano→mini is **+55% PR-AUC**, so
  there's rich *sequential* structure that capacity unlocks (LightGBM-on-aggregates, 0.043,
  can't see it).

**Most likely cause of the `small` dip = fixed-budget undertraining.** `small` has ~2×
`mini`'s params but the *same* 3000 steps. It has the **lowest MLM loss** (best pretrained
model) yet its frozen representation isn't the most linearly-separable for fraud — the
classic signature of a bigger model that's step-starved (lower pretraining loss doesn't
monotonically map to better linear-probe downstream when undertrained).

**Takeaway:** scaling helps a lot (nano→mini), but at a fixed 3000-step budget **`mini` is
the sweet spot**; `small` is probably step-starved, not capacity-capped. **Follow-up:**
compute-matched rerun (give `small` proportionally more steps) to find its true ceiling.
Caveats: single seed; `nano` uses L=96 vs L=128 for mini/small (minor confound).

---

## E6 — small, compute-matched (6000 steps): was it undertrained?

**Question:** E5's `small` dip — real ceiling or fixed-budget undertraining?

**Setup:** retrain `small` from scratch at **6000 steps** (2× E5's budget), cosine over the
full horizon; bucket + Δt; as-of-date probe. Repro:
`python -m pragma.train.pretrain --preset small --numeric-mode bucket --max-steps 6000 --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json --tag _dt_6k`

**Result:**

| model | ROC-AUC | PR-AUC | R@P0.5 | R@P0.9 | MLM loss |
|-------|---------|--------|--------|--------|----------|
| mini @3000 | 0.959 | 0.364 | 0.358 | 0.086 | 1.69 |
| small @3000 | 0.963 | 0.349 | 0.310 | 0.102 | 1.57 |
| **small @6000** | **0.977** | **0.495** | **0.523** | **0.193** | **1.39** |

**Interpretation — it was undertrained, decisively.** Doubling the budget lifts small's
PR-AUC **0.349 → 0.495 (+42%)**, well past mini, and Recall@P0.5 to **0.523** (catches >½
of all fraud at 50% precision). This resolves E5: the `small` dip was purely a fixed-step
artifact, not a capacity ceiling. Clean story now: **more parameters + enough training both
help**, and bigger models need proportionally more steps to convert capacity into
downstream signal (small's MLM loss kept falling to 1.39).

**Consequence for the scaling claim:** a step-matched sweep *understates* larger models.
The honest scaling law here needs each size trained to (near-)convergence; at 6000 steps
small is the clear best and likely not yet saturated. **Follow-ups:** push small further
(8-10k steps) to find the knee; bump mini to 6000 for an apples-to-apples mid-point.

Caveats: single seed.

---

## E7 — Pre-flight: does long-range fraud signal exist? (SIM headroom, data-only)

**Question:** before building SIM-style retrieval, does TabFormer fraud actually have
long-range structure that recency-128 misses? (Data-only, no model.)

**Method:** using past *labels* (upper bound on any retrieval), compare current-txn fraud
rate conditioned on whether prior fraud sits in the recent window vs the far tail (>128 ago).

**Result (full data, base rate 0.122%):**

| condition | P(fraud) | lift |
|-----------|----------|------|
| prior fraud within recent 128 | 3.95% | 32× |
| prior fraud only in far tail (>128 ago), none recent | 0.018% | 0.15× |
| no prior fraud at all | 0.019% | 0.16× |
| prior fraud at same merchant (any time) | 1.35% | 11× |
| prior *clean* visits to merchant (no prior fraud) | 0.037% | 0.3× |

**Interpretation — NEGATIVE, and decisive.** A card whose fraud is only in the distant
past behaves like one never defrauded (0.018% ≈ 0.019%, both below base). Fraud is a
**short-range burst process**: 32× sticky within recent-128, ~0 beyond it. Recency-128
already captures the signal; retrieving distant history would surface events with ~zero
fraud information. **Don't build SIM for this dataset/task** — recency truncation, which
looked naive, is well-matched because the phenomenon is local (consistent with the E4 Δt
win). Merchant-fraud stickiness (11×) is real but recent-dominated.

**Caveats:** (1) strong negative — measured with labels (the ceiling), so feature-based
retrieval can only do worse. (2) Specific to this *synthetic* dataset + *fraud*; real
banking data plausibly has long-range fraud (rings, dormant-reactivation), and other tasks
(LTV, credit, churn) likely depend far more on long history. So "SIM won't help here," not
"SIM is useless." The pre-flight saved a ~half-day build — the point of a pre-flight.

---

## E8 — Context length (seq len) vs PR-AUC: where does it plateau?

**Question:** how much recent history does the model actually need? Where does more
context stop helping?

**Setup:** train nano (bucket + Δt, 3000 steps) at context window **L ∈ {4,8,16,32,64,128,256}**,
as-of-date probe at the same L. Repro: `pretrain ... --max-seq-len L --tag _dt_L{L}` then probe.

**Result:**

```
   L  PR-AUC  %ofmax     ROC   R@P0.5   bar
   4   0.055     23%   0.935   0.000  #######
   8   0.151     64%   0.951   0.042  ###################
  16   0.167     71%   0.945   0.024  #####################
  32   0.212     90%   0.948   0.132  ###########################
  64   0.155     66%   0.944   0.057  ####################   <- single-seed dip
 128   0.227     97%   0.947   0.144  #############################
 256   0.235    100%   0.948   0.164  ##############################
```

**Interpretation — steep early gains, but NOT plateaued by 256.** PR-AUC rises sharply
from L=4→32 (90% of the L=256 value by L=32), *then keeps climbing with diminishing
returns*: 32 → 128 → 256 = 0.212 → 0.227 → 0.235, and **Recall@P0.5 rises monotonically to
256** (0.132 → 0.144 → 0.164). So most of the gain is early, but the slope stays positive
through L=256 — we have **not** located the plateau; it's beyond 256. ROC-AUC does saturate
early (~0.95 by L=8) — ranking quality is set by short-context features, while the extra
PR-AUC / high-precision recall past L=32 is the marginal payoff of more context.

Mild tension with E7 (which found no long-range *label* recurrence beyond 128): the 128→256
gain is small and likely comes from richer *behavioural baselines* (spend/merchant patterns,
familiarity as a negative signal) accumulating with more context, not from fraud-label
recurrence. Worth noting, not contradictory.

**The L=64 dip (0.155) is measurement noise — quantified.** Re-probing the *same frozen
L=64 checkpoint* 4× gives {0.155, 0.164, 0.194, 0.165} (true value ~0.17-0.19); re-probing
L=32 gives 0.180 (was 0.212), L=128 gives 0.228 (stable). The probe fits its logistic head
on a *random ~400-batch subsample* of train windows (unseeded), which alone swings PR-AUC by
**±0.02-0.04**. So L=32 (~0.19) and L=64 (~0.17) overlap — there's no real peak-then-valley;
the L=64 point just drew low. Only the broad trend (rise through 256) survives this noise;
per-point structure does not. Fix: seed the probe subsample (done) and average over
probe+training seeds for a clean curve.

**Caveats:** (1) nano is capacity-limited (~0.23 ceiling; small@6000 reached 0.495) — a
bigger model may keep gaining from context *longer*, so nano likely *understates* the
context benefit and the true knee could be further out. (2) tiling means "up to L" context
(avg ~L/2). (3) single seed.

**Takeaway (corrected):** more context keeps helping through L=256 (diminishing returns,
strongest at high precision); the plateau is **not yet reached**. To locate it, extend to
L∈{512,1024} — ideally seed-averaged and on a larger (non-capacity-limited) model.

---

## E9 — RoPE ablation: how much does the positional encoding matter?

**Question:** how much does RoPE contribute, and does keying it on *event time* beat plain
*ordinal* position — especially now that the Δt field also injects time (E4)?

**Setup:** nano, bucket + Δt, 3000 steps, as-of-date probe. `pos_mode ∈ {time, index, none}`,
**3 training seeds each**. Repro: `pretrain ... --pos-mode {time,index,none} --seed {0,1,2}`.

**Result (mean ± std over 3 seeds):**

| pos_mode | mean PR-AUC | std | seeds {0,1,2} |
|----------|-------------|-----|---------------|
| time (RoPE-on-time) | **0.164** | 0.057 | 0.227, 0.115, 0.151 |
| index (RoPE-on-index) | 0.140 | 0.074 | 0.220, 0.127, 0.074 |
| none (no RoPE) | 0.112 | 0.052 | 0.164, 0.060, 0.112 |

**Interpretation — directionally yes, statistically inconclusive.** The means order exactly
as theory predicts (time > index > none): RoPE helps (+0.052 vs none, ~+46% relative) and
time-keying beats ordinal. **But the seed variance swamps it** — std ~0.05-0.07, individual
runs span 0.06→0.23 *for the same config*, so time−none is only ~1.2 SEM at n=3. Can't claim
significance with 3 seeds.

**The bigger lesson (a caveat on earlier experiments).** nano@3000 is *extremely*
seed-sensitive (±~0.05 PR-AUC). The `time` arm's seed-0 draw (0.227) is essentially E4's
single-seed nano+Δt number (0.236), but the 3-seed mean is 0.164 — i.e. **E4 measured a
lucky-high seed, and its +0.017 Δt gain is smaller than the seed noise.** So:
- **Single-seed nano deltas (esp. E4) are not reliable** — treat as suggestive only.
- **Larger effects survive the noise and stand:** E3 (PLE/periodic ~0.11 below bucket),
  the scaling trend (nano→mini→small, E5), and small@6000 (0.49, E6).

**To actually resolve RoPE's contribution:** many more seeds, and/or run on a larger,
better-trained model (small@6000 was far more stable) which is less seed-sensitive. The
n=3 nano result is under-powered for a ~0.05 effect.

---

## E10 — Corrected evaluation (sliding-window as-of-date) + the L=256 eval-window collapse

**Why:** E1–E9 scored with method **(b)** — causal-masked *tiling* — which (i) under-contexts
events (~L/2 avg), (ii) applies a causal mask the *bidirectional* MLM model never trained with,
and (iii) isn't PRAGMA-faithful. Method **(a)**: for each target txn, a window *ending at it*,
run the frozen backbone **bidirectionally** (no future in the window), read the **last-position**
embedding — used for both probe-fit and scoring. This is exactly PRAGMA's "final `[EVT]` at the
evaluation point." **No retraining** — re-eval of saved checkpoints.

**Setup:** stratified test subsample (all frauds + 150k non-frauds → base rate **1.9%**), same
across all arms. The higher base rate lifts absolute PR-AUC, so compare arms **to each other**,
not to the old full-test (0.13%-base) numbers.

**Results (method a, base rate 1.9%):**

| arm | PR-AUC | R@P0.5 | R@P0.9 |
|-----|--------|--------|--------|
| LightGBM | 0.369 | 0.418 | 0.000 |
| PRAGMA nano | 0.643 | 0.711 | 0.268 |
| PRAGMA mini | 0.695 | 0.716 | 0.432 |
| **PRAGMA small (L128)** | **0.786** | 0.819 | 0.579 |
| small −Δt | 0.780 | 0.810 | 0.578 |
| small −RoPE | 0.714 | 0.753 | 0.401 |

**Corrected findings (vs the method-(b) versions):**
- **FFM ≫ LightGBM** (same subsample): small **0.786 vs 0.369 (+113%)**; R@P0.9 **0.58 vs 0.00**
  (LightGBM can't reach 90% precision at all). Headline holds, operating points stronger.
- **Scaling:** clean monotonic **0.643 → 0.695 → 0.786** (nano→mini→small).
- **Δt: +0.006 — marginal** (was +0.050 under (b)). With full-context windows the explicit Δt
  field adds ~nothing (the model gets timing from RoPE + full context). **E4 was distorted by (b).**
- **RoPE: +0.072** (was +0.115) — still clearly matters.

**The L=256 eval-window collapse (a real, instructive result).** Trained-and-scored at L, the
seq-len curve looked like a cliff — L64 0.792 / L128 0.786 / **L256 0.298**. But L256 *trained*
fine (best MLM loss 1.10, 0 skips). Diagnosis: re-scoring the **same** L256 checkpoint at shorter
windows recovers it — and it's the **best** backbone:

| L256 model, eval window | 64 | 128 | 256 |
|-------------------------|----|-----|-----|
| PR-AUC | **0.817** | 0.809 | 0.298 |

So the cliff is a **long-window readout pathology, not a capacity/context ceiling** — the same
checkpoint recovers to 0.81 at short windows. **Mechanism is a hypothesis, not proven:** most
likely **attention dilution** — the last-position readout aggregates over ~250 mostly-irrelevant
events (fraud is short-range, E7) — plausibly worsened by **RoPE-on-raw-time aliasing** (L256
windows span ~250 days; the top rotary frequency wraps every ~6 days), which would leave the model
unable to down-weight distant events by time. Wrinkle against pure aliasing: the model was
*trained* at L=256 yet fails at its native window and works at shorter ones — consistent with MLM
being a *local* task (best MLM loss 1.10 via local reconstruction) that never forced long-range
attention to be calibrated. **Isolating RoPE-vs-dilution is a TODO** (span-binned PR-AUC, or a
log-time / index-RoPE retrain).

**Takeaways:**
- **Train with generous context** (the L256 backbone read-short is the best, 0.817), but **score
  with a short window (~64)** — both because fraud is short-range (E7) *and* to avoid long-window
  aliasing.
- To use long eval windows, fix the time encoding: **log-time or learned-frequency RoPE**.
- The eval method genuinely mattered: (b) inflated Δt (+0.05 → +0.006) and would have mis-shown
  seq-len. (a) is correct and PRAGMA-faithful.

**Caveats:** single seed; subsample base rate 1.9% (absolute PR-AUC not comparable to the earlier
full-test numbers — relative comparisons only). Figures: `figures/fig{1..4}` regenerated from (a).

---

## E11 — Field-embedding ablation: is the per-field identity vector redundant?

**Question:** Each token in the Event Encoder is `value_emb(global_id) + field_emb(field)`, where
`field_emb` is a learned per-field identity vector (BERT segment-embedding analogue). But
`value_emb` is indexed by *global* id = `local_id + per-field offset`, so each field already
occupies a **disjoint block** of the value table — field identity leaks in through that side
channel. So does the explicit `field_emb` earn its keep, or is it redundant?

**Setup:** `small`, bucket + Δt, 6000 steps, seed 0, L=128 — *identical* to the E6/E10 baseline,
only `--no-field-emb` differs (flag added to `pretrain.py`; `field_emb` module kept in the
checkpoint but not added in the forward pass). Two signals: (1) as-of-date PR-AUC (method a, same
subsample); (2) **held-out val MLM loss** with fixed seed + fixed batch order + identical masking
(`scripts/eval_mlm.py`) — a much lower-variance signal than the downstream probe.

| field_emb | test PR-AUC (a) | ROC-AUC | R@P0.5 | val MLM loss |
|-----------|-----------------|---------|--------|---------------|
| **ON** seed 0 (baseline) | 0.7860 | 0.9807 | 0.819 | 1.447 |
| **ON** seed 1 | — | — | — | 1.252 |
| **OFF** seed 0 (ablated) | 0.7874 | 0.9841 | 0.836 | 1.282 |

**Result — `field_emb` is redundant / neutral (no real effect on either metric):**
- **Downstream: no effect.** ΔPR-AUC = +0.0014, well inside the ±0.005 seed band.
- **Pretraining: also no effect — and a methodology lesson.** The ablation's MLM loss (1.282)
  *first looked* like an improvement over the seed-0 ON baseline (1.447). But measuring a **second
  ON seed (1.252)** exposed that as an artifact: **MLM-loss seed variance is ~0.2**, *larger* than
  the apparent 0.165 "effect," and the OFF run (1.282) sits comfortably inside the ON range
  [1.252, 1.447]. The seed-0 ON baseline was simply an unlucky run. My initial "removing field_emb
  *helps* MLM" claim was noise and is **retracted**.

**Why neutral (as expected):** field identity is already carried — cleanly and for free — by the
value-table offsets (disjoint id blocks, *including* a per-field `[MASK]` row at `offset+1`, so even
masked cells keep their identity). The extra learned `field_emb` is a redundant constant the model
can trivially learn to ignore (a constant is subtractable → it *can't* hurt reconstruction at
convergence, and indeed doesn't). Confirms the prediction: small/zero effect, unlike RoPE (+0.072,
which had no side channel).

**Scope caveat (why this won't generalize):** redundancy here is a property of our **single,
dense, fixed-schema** event type — every transaction has the identical 11 fields in the same slots,
so the offset scheme fully determines identity. In a real FFM with **heterogeneous event types**
(payment / withdrawal / transfer / login, each a different field set) or **sparse/variable** fields,
you can't free-ride identity off a fixed offset layout, and the field/key embedding becomes
load-bearing. So E11 says "field_emb is free to drop *on TabFormer*," not "field embeddings are
useless in general."

**Caveats:** PR-AUC single-seed (null is safe, Δ ≪ noise). MLM loss now has 2 ON seeds + 1 OFF
seed; the cross-seed spread (~0.2) dwarfs any field_emb effect. An OFF seed 1 would complete the
2×2 but the within-noise conclusion is already clear. **Lesson: MLM loss is NOT low-variance across
seeds — never compare a single ablation run to a single baseline.** Artifacts:
`pretrain_small_bucket_dt_6k_nofe.pt`, `eval_small_bucket_dt_6k_nofe_swasof.json`.

---

## E12 — Tokenizing high-cardinality fields: merchant identity vs regional geo

**Question:** the high-cardinality fields (`merchant_name`, `merchant_city`, `zip`) are crc32-hashed
into 4,096 buckets, which destroys identity/geography (merchant_name collides ~24 merchants/bucket;
`La Verne`/`Monterey Park` hash to unrelated buckets). Two interventions, each a nano ablation:
(a) **unhash `merchant_name`** to a full identity vocab; (b) **replace `zip` with its 3-digit regional
prefix** as a categorical (849 regions — nearby zips share a prefix, giving coarse geography).

**Setup:** nano, bucket+Δt, L=96, 3000 steps, seed 0, as-of-date probe. Encoding overrides via new
`encode` flags `--mname-cat` / `--zip-prefix 3` (a `kind_overrides` hook in `Tokenizer.fit`). Eval on a
capped subsample (base rate **4.65%** — higher than the headline 1.9%, so PR-AUC here is only
comparable *across these three rows*, not to other experiments).

| arm | params | ROC-AUC | PR-AUC | R@P0.5 | R@P0.9 |
|-----|--------|---------|--------|--------|--------|
| baseline (mname+zip hashed) | 2.90M | 0.9529 | 0.7566 | 0.814 | 0.444 |
| (a) merchant_name unhashed (cat, 92k) | 19.96M | 0.9504 | **0.6553** | 0.769 | **0.083** |
| (b) zip → 3-digit regional prefix | **2.28M** | 0.9547 | **0.7697** | 0.820 | **0.517** |

**Result — geo helps (and is cheaper); naive identity hurts:**
- **(b) regional geo: +0.013 PR-AUC (+1.7%), +0.073 R@P0.5-9 (R@P0.9 +16%), with *fewer* params**
  (2.28M < 2.90M, since 849 cats < 4,098 hash buckets). Not a capacity gain — it's real representational
  signal from grouping nearby zips into learnable regions, and it concentrates at **high precision**
  (where fraud ops operate). Note: this is *categorical* embedding of the prefix (regional grouping),
  **not** distance-aware — the ordinal/lat-lon structure is still unexploited, so it's a **lower bound**
  on the geo lever.
- **(a) unhashing merchant_name: −0.10 PR-AUC (−13%), R@P0.9 collapses 0.44→0.08**, despite 7× params.
  The 92k embeddings are under-trained (rare merchants → noise), the 92k-way MLM reconstruction head is a
  near-impossible task that diverts the backbone, and hashing was acting as a useful regularizer. So
  *naive* identity encoding of a high-card ID field is **net-negative** at this scale/budget. (TabFormer's
  merchant_name is an obfuscated integer ID — no text/semantics to exploit; only identity, which needs a
  smarter recipe: identity *input* embedding but dropped from the MLM target, or sampled/hierarchical
  softmax.)

**Takeaway:** replacing random hashing of geographic fields with **region-preserving encodings** is a
cheap, principled win; blindly un-hashing a high-card ID field backfires. Best next step for geography is
true **lat/lon (Fourier/Sphere2Vec)** encoding of zip+city (needs an offline geocoding table).

**Caveats:** single-seed (the +0.013 PR-AUC is within a seed's noise, but the R@P0.9 jump + lower params
make the direction credible); capped-eval base rate 4.65%; (b) is coarse regional geo, not distance-aware;
(a)'s 92k table is under-trained so its number is a floor. Artifacts: `data/processed_{mnamecat,zipgeo}/`,
`eval_nano_bucket_dt_{mnamecat,zipgeo}_swasof.json`.
---

## E13 — Relational (cross-entity) signal: where to inject it — probe vs backbone

**Question:** PRAGMA encodes one `(user,card)` sequence at a time and is structurally blind
to **cross-entity** patterns (a merchant's recent activity across *all* cards). Does adding
causal **merchant** signal help fraud, and — the real question — does baking it into the
backbone as **cross-attention** beat just bolting the features onto the probe? (Full design
writeup: `RELATIONAL_PRAGMA.md`.)

### E13a — Relational features, complementarity (handcrafted, local)

Causal per-merchant features (prior txn count, prior fraud rate, prior mean amount,
card–merchant novelty), leakage-safe (strictly as-of-date), LightGBM in the probe's fit
regime. Standalone the merchant **prior fraud rate** carries real signal (ROC 0.847 / PR
0.314) — but it is *label-derived*. Complementarity on top of a strong within-sequence model:

| features | PR-AUC | ROC-AUC |
|---|---|---|
| within-sequence only | 0.881 | 0.991 |
| relational only | 0.356 | 0.923 |
| **within-sequence + relational** | **0.920** | 0.994 |

→ **+0.038 PR-AUC** from signal a per-sequence encoder cannot access. Real, but driven by the
label-based feature (see caveats + `RELATIONAL_PRAGMA §6`).

### E13b — Architectural test: entity-memory cross-attention vs the duct-tape

**Setup:** one controlled GPU run (RunPod **L40S**, bf16), `small`, bucket + Δt, **8000
steps**, batch 256, lr 4.24e-4, L=128, method-(a) probe on the standard stratified subsample
(n=152,928, base 1.9%). Two backbones trained **identically** — the *only* variable is where
the 5-dim causal merchant memory (`merchant_mem.npz`) enters:
- **ARM A (no-mem):** plain backbone → then features injected at the **supervised probe**
  (duct-tape concat), logreg and LightGBM heads (`scripts/fusion_probe.py`).
- **ARM B (memory-CSA):** merchant memory cross-attended by the History Encoder, trained
  end-to-end under the **same MLM objective**, read out by the **frozen linear probe**
  (`--mem`; `MemoryCrossAttention`).

| arm | where relational signal enters | probe head | PR-AUC | ROC-AUC | R@P0.5 | R@P0.9 |
|-----|-------------------------------|-----------|--------|---------|--------|--------|
| embedding-only (no-mem) | — | linear | 0.747 | 0.980 | — | — |
| **memory-CSA** (backbone, MLM) | **backbone** | linear | **0.694** | 0.975 | 0.758 | 0.312 |
| duct-tape fusion | **probe** (supervised) | logreg | 0.786 | 0.984 | — | — |
| duct-tape fusion | **probe** (supervised) | LightGBM | **0.844** | 0.988 | — | — |

**Interpretation — the elegant architecture LOST, and fell *below* baseline (−0.053).**
Injecting relational signal into the MLM-pretrained backbone *hurt*; injecting the same
features at the supervised probe won by +0.09 (logreg) to +0.15 (LightGBM). **Where you
inject matters more than how elegant the injection is:**
1. **Objective mismatch (the mechanism).** The backbone trains on **MLM** (token
   reconstruction). The strongest relational feature — merchant prior fraud rate — is
   *label-derived* and useless for reconstructing masked tokens, so MLM gives **no gradient**
   to preserve it; the cross-attention pathway optimises for reconstruction, not fraud.
2. **Frozen-backbone bottleneck.** Any captured signal must survive as a *linearly-readable*
   direction in the frozen embedding — lossy vs handing the raw feature to the probe.
3. **Cost without benefit.** +38k params / a new route on the same step budget, mildly
   perturbing the representation with nothing to compensate.

This answers a standing open question ("does relational signal help the *linear-probe*
readout, or need fine-tuning?") for the pretraining-injection route: **under a frozen backbone
+ MLM, no — inject downstream.**

**Caveats — not yet a settled negative.**
1. **Absolute numbers ≠ the canonical baseline.** This run's config (8k / batch 256 / bf16 /
   sqrt-scaled lr) differs from E10's canonical small (6k / batch 128 / fp32 → 0.786), so
   embedding-only reads **0.747** here. The *within-run* comparison is valid (all four arms
   share this exact backbone); the cross-run absolute is not — don't cite 0.747 as "the" small
   number.
2. **Single seed**, one dataset. The Δ(memory-CSA − baseline) = −0.053 clears probe noise
   (±0.005) but not necessarily *training-seed* noise (±~0.05 at nano; smaller at small but
   unmeasured here).
3. **Untested rescues:** longer pretraining; a **relational auxiliary objective**
   (masked-entity-context / contrastive co-occurrence, `RELATIONAL_PRAGMA §4`) so MLM *rewards*
   the memory; light fine-tuning of the mem pathway; a dataset with **real label-free**
   relational fraud (TabFormer fraud is rule-injected per-user → thin cross-entity structure,
   so this **understates** the real-world upside).

**Repro:**
`python scripts/build_merchant_memory.py --parquet data/processed/transactions.parquet --data-dir data/processed_dt`
then ARM A: `pretrain --preset small --numeric-mode bucket --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json --max-steps 8000 --batch-size 256 --lr 4.24e-4 --dtype bf16 --tag _A`
→ `python scripts/fusion_probe.py --ckpt artifacts/pretrain_small_bucket_A.pt --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json --parquet data/processed/transactions.parquet`;
ARM B: add `--mem --tag _B` to pretrain → `python -m pragma.train.asof_probe --ckpt artifacts/pretrain_small_bucket_B.pt --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json`.
Full metrics: `artifacts/relational_8k_results.json`. Feature flag: `use_mem` (default off).

---

## Engineering notes

- **Timing (M4 Max, MPS):** nano ≈ 0.48 s/step (~13 min/epoch); small ≈ 2.97 s/step
  (~60 min/epoch). Full runs feasible locally; no remote compute needed at this scale.
- **MPS NaN fix:** a `finfo.min` additive attention mask caused NaN in MPS softmax-backward
  mid-training. Fixed with a moderate finite mask (`-1e4`) + a guard that skips the optimizer
  step on any non-finite gradient. Bucket runs then train clean (0 skips); the new PLE/
  periodic Linear layers show ~5-7% early skips (benign, guard-absorbed, conservative).

## Open questions / next experiments

- [ ] ~~SIM-style retrieval~~ **de-prioritised for this dataset (see E7):** the pre-flight
  found no long-range fraud signal — fraud is a short-range burst process, so retrieving
  distant history won't help *TabFormer fraud*. Revisit only for a dataset/task with genuine
  long-range dependence (real banking data; LTV/credit/churn). The hard-truncation limitation
  (scoring on the last ~4% of history) stands, but here it's benign.
- [ ] Seed sweep (≥3 seeds) to put error bars on E3/E4/scaling deltas.
- [ ] Scaling: if `small` looks flat, rerun compute-matched (more steps for bigger models)
  — the current sweep is fixed at 3000 steps, so a flat point may be undertraining.
- [ ] E3 fair rematch: continuous/regression MLM target for the numeric field.
- [ ] LoRA fine-tuning as a second PRAGMA readout (paper reports this too).
- [ ] From-scratch task-specific transformer (the paper's literal "domain-specific" contender).
- [ ] Temporal-split variant (harder, shift-confounded) for contrast.
- [ ] Log-transform / learned frequencies for RoPE time input (aliasing at large gaps).
- [x] Field-embedding ablation (E11): **neutral** — redundant given the value-table offsets, no
  real effect on PR-AUC or MLM loss (an apparent MLM "drop" was seed noise; MLM-loss seed variance
  ~0.2). Won't generalize to heterogeneous/sparse-schema FFMs.
- [x] Relational / cross-entity signal (E13): merchant features add **+0.038** on a strong
  within-seq model (E13a); but **architectural** entity-memory cross-attention **lost to** the
  duct-tape probe fusion and fell below baseline (E13b) — inject relational signal at the
  *supervised probe*, not the *MLM backbone*.
- [ ] Relational rematch: pretrain memory-CSA with a **relational auxiliary objective** (so MLM
  rewards the memory) and/or light-finetune the mem pathway; re-test on data with real label-free
  relational fraud. The E13b negative may not survive these.
- [ ] Untuned defaults worth a cheap sweep: amount buckets (16/64/256), hash size (4096 →
  16k/32k, esp. merchant_name which collides ~24 merchants/bucket), merchant_city semantic/geo
  embedding (real text currently hashed away).

