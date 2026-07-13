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
