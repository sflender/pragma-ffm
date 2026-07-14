# Experiment Log — IEEE-CIS (Vesta)

Second dataset for the FFM, chosen because — unlike TabFormer's synthetic per-user injected
fraud — IEEE-CIS is **real card-not-present e-commerce fraud** with genuine cross-entity
structure (many card sequences share `addr1` / `P_emaildomain` / `card2`). The goal is to test
whether the FFM's findings hold where the fraud is real, and eventually whether relational
modelling (which failed on synthetic TabFormer, `RELATIONAL_PRAGMA §6.1`) pays off here.

Shares the exact model/train/probe code with TabFormer via the dataset-adapter refactor
(`pragma.data.schema` + `pragma.data.ieee_cis`); only the field schema and parser differ.

## Fixed setup

- **Data:** IEEE-CIS `train_transaction` ⋈ `train_identity` (590,540 txns, **3.5% fraud** —
  ~29× denser than TabFormer). Source: Kaggle competition `ieee-fraud-detection`.
- **Sequence entity:** `card1` (13,553 cards, mean 44 txns; cards with ≥5 txns cover 97.7% of
  rows), ordered by `TransactionDT`. `ts = 2017-12-01 + TransactionDT` → real hour/day-of-week.
- **Fields (16):** the *interpretable* core only — amount, ProductCD, card4/6/2/3/5, addr1/2,
  P/R_emaildomain, DeviceType, hour, dow, DeviceInfo (hash), dt. **The 400+ anonymised Vesta
  `V*`/`C*`/`D*`/`M*` engineered columns are deliberately dropped** (both arms; see caveats).
  Tokenizer: F=16, V=5448.
- **Split:** per-`card1` sequence, seed 0, 80/10/10 (unseen-card generalisation, as TabFormer).
- **Eval:** as-of-date sliding-window probe on a stratified test subsample (all fraud + capped
  non-fraud), base rate **3.0%**. PR-AUC headline. Same harness/subsample for both arms.

---

## I1 — Moonshot: does the FFM beat LightGBM on real (IEEE) fraud?

**Question:** on TabFormer the frozen MLM-pretrained FFM **beat** the GBDT incumbent (E1/E10).
Does that hold on real e-commerce fraud, restricted to the same interpretable fields for both?

**Setup:** pretrain mini-PRAGMA (MLM, bucket + Δt, bf16) on IEEE `card1` sequences, then a frozen
last-position embedding → logistic probe. LightGBM on the **same 16 fields** as a causal flat
table + within-card aggregates (`pos_in_seq`, `dt_last`, causal expanding amount mean). Both
scored on the identical as-of-date test subsample. Repro: `scripts/ieee_probe_vs_lgbm.py`.

| arm | params | steps | PR-AUC | ROC-AUC |
|-----|--------|-------|--------|---------|
| **LightGBM (16 fields + causal aggs)** | — | — | **0.157** | **0.754** |
| FFM small (frozen probe) | 13.7M | 6000 | 0.106 | 0.726 |
| FFM nano (frozen probe) | 2.9M | 3000 | 0.090 | 0.706 |

**Result — the FFM LOSES to LightGBM (−0.05 PR-AUC), the opposite of TabFormer.** Bigger/longer
helps the FFM (nano 0.090 → small 0.106) but not enough to catch the trees.

**Interpretation — the FFM's edge is dataset-dependent, and this is *why*.**
- **TabFormer fraud is sequential; IEEE fraud is static-tabular.** TabFormer fraud is a burst
  process — stolen cards fire several txns seconds apart (E7: 32× stickiness within recent
  events), exactly the structure a sequence encoder captures, so the FFM won there. IEEE
  card-not-present fraud is far more a function of the **current transaction's own attributes**
  (device, email domain, product, amount) than of the card's recent history — a per-row tabular
  pattern, which gradient-boosted trees model directly.
- **A frozen MLM backbone is structurally disadvantaged for static signal.** The FFM must
  compress history into an embedding a *linear* probe then reads; LightGBM's trees split on the
  raw current-txn fields with no bottleneck. When the signal is "this transaction's feature
  combination" rather than "this card's behaviour over time," the trees win. (Same
  proxy-alignment logic as `RELATIONAL_PRAGMA §7.1`: MLM optimises reconstruction of the 16
  fields, not the fraud-relevant feature interactions the probe needs.)
- **Consistent story across datasets:** the FFM helps when the task signal is *sequential and
  MLM-aligned* (TabFormer), and underperforms trees when it is *static and per-transaction*
  (IEEE). That's a sharper, more honest claim than "FFMs beat GBDTs."

**Caveats — do not over-read this as "FFMs lose on real data."**
1. **Thin field subset.** Both arms use only 16 interpretable fields; IEEE's signal largely
   lives in the 400+ Vesta `V*` features we dropped. Real leaderboard solutions hit ROC ~0.96.
   This is a *sequence-encoder-vs-trees on matched fields* test, not a capability ceiling.
2. **Likely undertrained.** small ran 6k steps; on TabFormer small needed ~15k to converge
   (E14). MLM loss was still falling. A longer run is the first thing to try.
3. **Frozen linear probe**, single seed, un-tuned (batch/lr copied from TabFormer). LoRA or
   light fine-tuning could change the verdict.
4. **The relational question is untouched here.** This is baseline FFM vs trees; the entity
   (`addr1`/`P_emaildomain`) memory experiment — the actual reason we came to IEEE — is next.

**Cost/repro:** ~$1.4 on one L40S (nano 600s + small 4204s + probes). Kaggle→parse→encode ran
on-pod in 28s. Artifacts: `artifacts/ieee_moon_{nano,small}.json`.

**Next:**
1. Train small **to convergence** (≥15k) + tune lr/batch for IEEE; re-check the gap.
2. Add a slice of the Vesta `V*` features (bucketed) to both arms — does more signal favour
   either architecture?
3. **The relational experiment:** entity-memory (`addr1`, `P_emaildomain`) cross-attention vs
   duct-tape fusion — does real cross-entity fraud rescue the architectural approach that lost
   on synthetic TabFormer (`RELATIONAL_PRAGMA §6.1`)? This is the reason IEEE is here.

---

## I2 — Client entity + longer sequences (do the FFM's own levers help?)

**Question:** the moonshot FFM (I1) lost to LightGBM. Two natural levers: a **cleaner client
entity** (`card1` is a noisy client; `card1+addr1` is closer to a real client) and **longer
sequences** (`card1` histories run to ~15k txns; L=128 truncates). Do either close the gap?

**Setup:** `small`, bucket+Δt, bf16, 8k steps; `scripts/ieee_probe_vs_lgbm.py` (FFM frozen
probe vs causal-tabular LightGBM on the *same* target subsample). New `--seq-key` option.

| arm | entity / L | base | FFM PR / ROC | LGBM PR / ROC | FFM−LGBM (PR) |
|-----|-----------|------|--------------|---------------|---------------|
| baseline (I1) | card1 / 128 | 0.030 | 0.106 / 0.726 | 0.157 / — | −0.051 |
| **A: client entity** | card1+addr1 / 128 | 0.030 | **0.115** / 0.729 | 0.152 / 0.735 | **−0.037** |
| B: longer sequences | card1 / 256 | 0.042 | 0.149 / 0.740 | 0.228 / 0.790 | **−0.079** |

**Result:**
- **Client entity helps (modestly).** Arm A shares the baseline's base rate, so it's directly
  comparable: FFM 0.106→**0.115**, gap −0.051→**−0.037**. `card1+addr1` is a cleaner client
  *and* gives 3× more training sequences (39,974 vs 13,553), easing data-starvation. Directionally
  confirms the entity lever.
- **Longer sequences do NOT help the FFM.** Arm B's gap *widens* to −0.079: LightGBM exploited
  longer `card1` histories more (its causal aggregates get richer with length), while the FFM
  didn't keep pace — and L=256 risks the RoPE long-window aliasing seen on TabFormer (E8/E10).

**Caveat (methodological):** the `--seq-key` change re-factorised `card1` (int→string), reshuffling
the per-card split, so Arm B lands on a different test partition (base **0.042** vs 0.030). PR-AUC
scales with base rate, so **only the within-arm FFM−LGBM gap is comparable across rows**, not the
absolute PR. Fix for future runs: pin the split independent of factorisation order. Also L=256
crashed once on the B×L=65536 SDPA kernel limit — fixed by capping batch to 128 at L=256.
Artifacts: `artifacts/ieee_moon_{A_card1addr1,B_card1_L256}.json`.

## I3 — Why is even LightGBM weak? Feature ablation (the real diagnosis)

**Question:** a proper IEEE-CIS model scores ROC ~0.96; our LightGBM sits at ~0.74–0.79. Bug, or
setup? Local diagnostic — LightGBM across feature sets × splits (300 rounds, full test split):

| features | split | ROC | PR-AUC |
|----------|-------|-----|--------|
| **full (~430)** | random | 0.963 | **0.771** |
| full (~430) | temporal (competition-like) | 0.914 | 0.553 |
| full (~430) | per-card (ours) | 0.883 | **0.586** |
| **16 fields (ours)** | per-card | 0.809 | 0.238 |
| 16 fields (ours) | temporal | 0.828 | 0.251 |

**Findings:**
1. **The dropped Vesta features are the dominant lever — ~0.35 PR-AUC.** Full-feature LGBM on our
   exact per-card split: **PR 0.586**; our 16 interpretable fields: **PR 0.238**. The 400+ `V/C/D/M`
   engineered columns (entity-relation aggregations, counts, timedeltas) carry most of the signal.
2. **The per-card split is fine** — it scores *higher* than the temporal split (0.586 vs 0.553), so
   our unseen-client methodology isn't the problem.
3. **The moonshot's own LGBM (0.152) was under-powered** vs a clean 16-field LGBM (0.238): it fits
   on the rebalanced target subsample (~10% base) then scores at 3% base, hurting PR calibration.
   Fix: train LGBM on the natural distribution.

**Everything in I1/I2 ran in a signal-starved toy regime.** The FFM-vs-trees question was never on a
fair footing; features (and relational structure) dwarf the entity/length deltas.

### Discussion — why the FFM can't *learn* the V-features (yet)

Ideally the FFM learns these representations from raw sequences, making Vesta's engineering
redundant. It fails here for four reasons, only some fundamental:
1. **Architecturally blind to cross-entity signal (the ceiling).** The `V*` features are mostly
   *cross-entity* aggregations (how many cards/emails/addresses tie together, network velocity),
   computed with global data access. Our FFM attends **within one `card1` sequence** — other
   entities are outside its receptive field, so it *cannot* learn them regardless of training. A
   per-sequence model has a hard ceiling below full-feature trees.
2. **MLM doesn't reward the aggregation** (proxy-alignment gap, §7.1): reconstructing a masked field
   never requires computing fraud-relevant counts/ratios.
3. **Frozen backbone + linear probe** can't recompute nonlinear aggregates it didn't surface.
4. **Data starvation** — learning aggregation functions needs scale we lack (~8k windows).

Within-sequence temporal signal (velocity/recency, overlapping `D`/`C`) the FFM *can* learn and
partly does; the **cross-entity** bulk of `V` is out of reach for a per-sequence model. So making
the V-features redundant requires a **relational** FFM (cross-entity memory / attention) + an
**aligned objective** (relational SSL or fine-tuning) + **scale** — which is exactly the
entity-memory experiment. That reframes it from "nice ablation" to the load-bearing test.

---

## I4 — Learned entity-memory (`addr1`): does cross-entity signal help on real data?

**Question:** the load-bearing relational test, mirroring TabFormer E13 on *real* cross-entity
fraud. Sequence on the client (`card1`); give the model a causal **memory over `addr1`** (billing
region, shared across cards). Does injecting cross-entity signal — as a duct-tape probe feature or
architecturally (memory cross-attention) — close the gap? On synthetic TabFormer, memory-CSA *lost*
(§6.1); the prediction was that *real* relational fraud might reverse it.

**Setup:** `small`, bucket+Δt, bf16, 8k steps, L=128, seq=`card1`; 5-dim causal `addr1` memory
(`build_merchant_memory.py --entity addr1 --card-col card1`). All arms share one split (base 0.042),
so the four-way is internally comparable. Repro: `fusion_probe.py --entity addr1` (Arm A no-mem) +
`asof_probe` on the `--mem` backbone (Arm B).

| arm | PR-AUC | ROC-AUC | R@P0.5 |
|-----|--------|---------|--------|
| relational-only (`addr1` feats) | 0.124 | 0.705 | — |
| memory-CSA (backbone, MLM) | 0.161 | **0.762** | 0.027 |
| embedding-only (no-mem) | 0.166 | 0.741 | — |
| **duct-tape fusion (logreg)** | **0.171** | 0.747 | — |
| duct-tape fusion (LGBM) | 0.165 | 0.757 | — |

**Result — the single-entity memory barely moves PR-AUC, and memory-CSA loses again.**
- **Duct-tape adds ~nothing** (+0.005 logreg, −0.001 lgbm over embedding-only) — vs +0.04–0.10 on
  TabFormer. **memory-CSA (0.161) sits *below* embedding-only (0.166)** and below the duct-tape —
  **replicating the TabFormer E13 negative on real data.**
- **Nuance worth keeping:** memory-CSA has the **best ROC (0.762 vs 0.741)** but the worst
  high-precision recall (R@P0.5 0.027). So the `addr1` memory *does* improve overall ranking — it
  captures some real cross-entity signal — but it **hurts the high-precision tail** PR-AUC rewards.

**Why the `addr1` memory is too weak (and what it implies):**
1. **`addr1` is already an input field** — a 5-feature aggregate over it is largely redundant with
   what the FFM embedding already encodes. TabFormer's `merchant_name` memory was stronger partly
   because merchant identity was *hashed away* in the base tokenizer.
2. **IEEE's cross-entity signal is multi-entity.** The Vesta `V*` features (I3, worth ~0.35 PR-AUC)
   aggregate over `card+addr+email+device` combinations; a single `addr1` memory is a thin proxy.
3. **K=1 compression is lossy.** Our memory is a single summary vector per entity — the degenerate
   K=1 case of a full **cross-sequence ("third-transformer") encoder** that would attend to the last
   K causal entity-neighbours. And it carries the same proxy-alignment gap (MLM doesn't reward using
   it), so architectural injection underperforms probe injection, as on TabFormer.

**Takeaway — consistent across both datasets:** bolting a *single, hand-picked* entity memory onto
the FFM (architecturally or via duct-tape) does **not** recover the cross-entity signal. The thesis
(FFM supersedes hand-engineered relational features) needs **rich multi-entity cross-sequence
attention** (last-K neighbours over `addr1`+`card2`+email) **plus a relational pretraining objective**
that rewards using it — not a K=1 summary under pure MLM. That is the concrete next architecture.
Artifacts: `artifacts/ieee_{fusion_A,memcsa_B}_addr1.json`.
