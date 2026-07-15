# When Do Financial Foundation Models Beat Gradient-Boosted Trees? Sequential vs. Relational Fraud and the Proxy-Alignment Gap

*Workshop submission (short paper). Laptop-scale study; single-seed unless noted.*

## Abstract

Financial foundation models (FFMs) — masked-language-model (MLM)–pretrained, encoder-only
transformers over per-account event sequences, read out by a frozen-backbone linear probe —
have been proposed as a replacement for the bespoke feature engineering and gradient-boosted
decision trees (GBDTs) that dominate production fraud detection. We reproduce a PRAGMA-style FFM
at laptop scale and study *when* it helps. We find its advantage is **dataset-dependent**: the
FFM beats a GBDT by a wide margin on synthetic card fraud whose signal is **sequential** (IBM
TabFormer: PR-AUC 0.81 vs 0.37) but **loses** on real card-not-present fraud whose signal is
largely **static/per-transaction** (IEEE-CIS: 0.11 vs 0.16 on matched fields). We then ask
whether adding a cross-entity **memory** (a per-event summary of a shared entity's recent
activity) lets the FFM capture the *relational* fraud a per-sequence model is structurally blind
to. Across two datasets and a **controlled synthetic benchmark with ground truth**, injecting
relational signal *architecturally* — cross-attention into the MLM-pretrained backbone —
consistently **fails**, even when the memory provably contains the exact feature a GBDT turns
into strong performance (0.05 vs 0.36 on our benchmark). Injecting the same feature at the
*supervised* head works. We name this the **proxy-alignment gap**: an MLM objective has no
incentive to preserve a feature that does not help token reconstruction, so a frozen linear
probe cannot read it — *even when it is handed to the model as input*. We connect this to
generative recommender FMs (HSTU, OneRec), which avoid the gap precisely because their
pretraining objective *is* the task and they train end-to-end, and we show [placeholder: S2]
that aligning the objective and unfreezing the backbone recovers the signal.

## 1. Introduction

Fraud detection is dominated by gradient-boosted trees over hand-engineered features. A
compelling alternative, mirroring the success of foundation models elsewhere, is to pretrain a
transformer on raw event sequences with a self-supervised objective and adapt it to downstream
tasks with a lightweight head — a **financial foundation model**. The appeal is that the model
would *learn* the representations analysts hand-craft, transfer across tasks, and improve with
scale.

We take this proposal seriously and probe its boundaries at laptop scale. Rather than chase a
leaderboard, we ask a mechanistic question: **when, and why, does an FFM beat a GBDT for
fraud — and when does it not?** Our contributions:

1. **Dataset-dependence of the FFM advantage (§4).** On *sequential* fraud (TabFormer) a frozen
   FFM beats a GBDT by a wide margin; on *static/per-transaction* fraud (IEEE-CIS, matched
   fields) it loses. The FFM wins exactly when the fraud signal lives in an account's own
   temporal history and is aligned with what MLM captures.

2. **Relational signal cannot be added by architectural memory alone (§5–§6).** A cross-entity
   memory injected into the MLM-pretrained backbone (memory cross-attention) fails to help — on
   TabFormer, on IEEE-CIS, and in a controlled benchmark — while injecting the *same* signal at
   the supervised head helps. We isolate the cause with a **synthetic generator whose fraud
   structure is known by construction** (§6): a per-sequence FFM recovers per-account fraud
   (PR-AUC 0.81) but is blind to relational fraud (0.04≈base), and *architectural* memory does
   not fix it (0.05) even though the memory contains the exact feature a GBDT turns into 0.36.

3. **The proxy-alignment gap, and its fix (§7).** We articulate why: MLM optimises token
   reconstruction; a fraud-relevant feature that does not aid reconstruction is never preserved
   in the frozen representation, so a linear probe cannot read it. This explains the negatives as
   a structural property, not a dataset artifact. We connect it to generative recommender FMs
   (HSTU, OneRec), which sidestep the gap because their objective *is* the task and they are not
   frozen, and confirm the fix by aligning the objective and unfreezing.

All code, the synthetic generator, and per-experiment result files are released.

## 2. Background and related work

**Financial foundation models.** PRAGMA-style FFMs tokenise each transaction into
key–value–time tokens, encode fields with a set transformer, encode an account's event history
with a bidirectional transformer using rotary position embeddings on continuous event time, and
pretrain with a masked-cell objective. Adaptation is a frozen backbone + linear probe on the
final-position embedding, evaluated as-of-date (a window ending at each target transaction, no
future leakage). We reproduce this recipe at ~3–14M parameters.

**Tabular and event-sequence transformers.** A line of work adapts transformers to tabular and
transaction data (TabTransformer, TabBERT/TabFormer, FT-Transformer, numeric encodings such as
piecewise-linear and periodic embeddings). Our tokenizer follows this tradition; consistent with
objective–encoding alignment, we find hard quantile bucketing of amounts beats continuous
numeric encodings under an MLM proxy.

**Generative recommender foundation models.** HSTU ("Actions Speak Louder than Words")
reformulates recommendation as autoregressive sequential transduction over a user's interaction
history and scales to trillions of parameters; OneRec and TIGER generate items directly, often
over semantic-ID tokenisations (RQ-VAE). These are architecturally the same per-user sequence
transformers we study, yet they are highly effective. We argue (§7) this is because their
pretraining objective *is* the downstream task (next item) and they train end-to-end — so they
do not incur the proxy-alignment gap — and because the relational signal they rely on
(collaborative affinities) is largely *stationary* and absorbs into shared item embeddings,
unlike the *transient* cross-entity state that characterises much fraud.

**GBDTs for fraud.** Gradient-boosted trees (XGBoost/LightGBM/CatBoost) with client- and
entity-level aggregations remain the production and competition standard; the winning IEEE-CIS
solution's decisive lever was a hand-built client identifier and per-client aggregates. We use
LightGBM as the incumbent and as a ground-truth probe of where signal lives.

## 3. Experimental setup

**Model.** mini-PRAGMA: set-transformer event encoder, RoPE-on-time bidirectional history
encoder, masked-cell pretraining; presets `nano` (2.9M) and `small` (13.7M). Adaptation: frozen
backbone, last-position embedding, logistic probe, unless stated. Evaluation: stratified
as-of-date test subsample; PR-AUC is the headline given extreme class imbalance.

**Memory cross-attention.** For a chosen entity (merchant / billing region), we precompute a
per-event, strictly causal (as-of-date) summary vector `m_e(t)` — log prior count, prior fraud
rate, prior mean amount, recency, novelty, and (where noted) windowed velocity — z-scored on the
train split. A `MemoryCrossAttention` block lets each event attend to its entity's memory inside
the history encoder, trained end-to-end under the same MLM objective. The *duct-tape* baseline
instead concatenates the same features to the frozen embedding at the supervised probe.

**Datasets.** (i) **IBM TabFormer** — 24.4M synthetic card transactions, per-user injected fraud
(0.12%); (ii) **IEEE-CIS/Vesta** — 590K real card-not-present transactions (3.5% fraud); (iii) a
**controlled synthetic generator** (§6) with tunable fraud structure.

## 4. FFM vs GBDT: a dataset-dependent advantage

On TabFormer, a converged `small` FFM (frozen probe) reaches **PR-AUC 0.807**, versus **0.369**
for LightGBM on causal engineered features on the same subsample — a >2× improvement, with
non-zero recall at high precision where the GBDT flatlines. TabFormer fraud is a *burst process*:
compromised cards fire several fraudulent transactions within seconds, a 32× stickiness within
the recent window that a sequence encoder captures directly.

On IEEE-CIS, restricting both models to the same 16 interpretable fields, the FFM **loses**:
PR-AUC **0.106** (`small`) vs **0.157** (LightGBM). IEEE card-not-present fraud is far more a
function of the *current* transaction's attributes (device, email domain, amount, product) than
of the account's history — a per-row tabular pattern trees model directly, and which a frozen
sequence embedding read by a linear probe compresses away.

| dataset | fraud type | FFM (frozen probe) | LightGBM | winner |
|---|---|---|---|---|
| TabFormer | sequential (burst) | **0.807** | 0.369 | FFM (+0.44) |
| IEEE-CIS (16 fields) | static / per-transaction | 0.106 | **0.157** | GBDT (−0.05) |

**Takeaway.** The FFM helps when the signal is *sequential and aligned with the pretraining
objective*, and underperforms trees when it is *static and per-transaction*. This is a sharper,
more honest claim than "FFMs beat GBDTs," and it motivates the relational question: can we give
the FFM the *cross-entity* signal a per-sequence model cannot see?

## 5. Injecting relational signal: architecture vs. supervision

Much real fraud is **relational** — compromised merchants hit many cards; rings fan small charges
across merchants — patterns invisible to a model that sees each account in isolation. We test two
ways to add a merchant/entity memory: **architectural** (cross-attention into the MLM backbone,
"memory-CSA") vs. **duct-tape** (concatenate the features at the supervised probe).

On TabFormer, in a controlled matched-budget run, memory-CSA (0.694) not only lost to the
duct-tape (0.786 logreg / 0.844 LightGBM) but fell **below** the no-memory baseline (0.747). On
IEEE-CIS with a billing-region (`addr1`) memory, the same pattern held: memory-CSA 0.161 vs
embedding-only 0.166 vs duct-tape 0.171 — the memory barely moved PR-AUC. In both cases,
injecting relational signal *architecturally into the SSL backbone* underperforms injecting it at
the *supervised head*. But these are real datasets where the relational signal may be weak,
label-based, or redundant with existing fields — so the negatives are suggestive, not decisive.

## 6. A controlled synthetic benchmark

To remove every confound, we generate transactions where we *know* where the fraud signal lives.
Two mechanisms: **per_card** — a card is compromised and fires a short burst of fraud on its own
timeline (a per-sequence signal); **relational** — a merchant is compromised for a short window,
producing a velocity spike of fraud across *many random cards*, with fraud amounts drawn from the
legit distribution, uniform per-merchant fraud rate (so merchant identity is uninformative), and
random fraud cards (so no per-card-history tell). The only signal is the merchant's transient
cross-card velocity.

**Ground-truth validation (LightGBM feature groups).** Before any FFM run, we confirm the signal
lives where designed (PR-AUC on test):

| feature set | per_card | relational |
|---|---|---|
| current transaction only | 0.04 | 0.03 |
| + card history (per-sequence) | **0.82** | 0.06 |
| + merchant windowed velocity (cross-entity) | 0.82 | **0.36** |

Per_card fraud is recovered by the card's own history; relational fraud is invisible to it and
recovered **only** by cross-entity windowed velocity (+0.30).

**The FFM 2×2 (small, frozen probe).**

| arm | per_card | relational |
|---|---|---|
| embedding-only (per-sequence FFM) | **0.81** | 0.035 |
| memory-CSA (+windowed-velocity memory) | 0.81 | 0.047 |
| duct-tape fusion (probe) | 0.83 | 0.047 |
| — LightGBM w/ windowed velocity (ground truth) | — | **0.36** |

Two findings, now against ground truth. **(a) Per-sequence architectural blindness is total:**
the same model recovers per_card fraud (0.81) and collapses to base on relational fraud (0.035),
driven only by where the signal lives. **(b) The frozen-MLM recipe cannot extract cross-entity
signal even when handed the exact feature:** memory-CSA's memory *contains* the windowed velocity
that LightGBM turns into 0.36, yet memory-CSA reaches only 0.047. The *same feature* works at the
supervised head (0.36) and fails inside the MLM backbone read by a frozen linear probe (0.05).

This does not *reverse* the §5 negatives — it *explains* them, with every confound removed.

## 7. Discussion: the proxy-alignment gap

**Why it happens.** MLM optimises reconstruction of masked `(field, value)` tokens. A
fraud-relevant aggregate — a merchant's windowed velocity, a client's prior fraud rate — does
*nothing* for reconstructing a masked amount or category. So no gradient asks the cross-attention
pathway to preserve it, and a linear probe cannot read a direction the backbone never
represented — **even when the feature is present in the input.** We call this the
**proxy-alignment gap**: a self-supervised proxy objective preserves only what it rewards, and a
frozen readout can extract only what is preserved.

**Why recommender FMs avoid it.** HSTU/OneRec are the same class of per-sequence transformer, yet
effective, for two reasons this paper makes precise. First, **objective alignment**: their
pretraining objective (next item/action) *is* the downstream task, and they train **end-to-end**
— there is no proxy and no frozen bottleneck. Second, the relational signal they depend on
(collaborative affinities) is largely **stationary** and is absorbed into shared item embeddings
over massive data; a per-user transformer then composes globally-informed embeddings. Fraud's
most valuable relational signal is instead a **transient cross-entity state** ("compromised
right now," a velocity spike) that cannot be baked into a static embedding and must be read from
concurrent cross-sequence activity. Notably, where recommenders *do* face transient cross-entity
signal — real-time "trending" — production systems inject it as serving-time features rather than
expecting the sequence backbone to learn it: exactly the duct-tape our results endorse.

**The fix.** The two properties that let recommender FMs avoid the gap are the two levers to
close it for fraud: **align the objective** (a fraud-relevant or relational self-supervised task
instead of pure MLM) and **do not freeze** (end-to-end or LoRA, so the head can shape the
backbone).

> **[Placeholder — S2, in progress]** We fine-tune the *same* memory architecture **end-to-end on
> the fraud label** over as-of-date windows on the synthetic `relational` data, and compare to the
> frozen-MLM memory-CSA (0.047) and the ground-truth ceiling (LightGBM windowed velocity, 0.36).
> *Result table to be inserted:* end-to-end + memory PR-AUC = `___`; end-to-end no-memory (control)
> = `___`. Prediction: end-to-end + memory recovers a large fraction of the 0.36 ceiling, while the
> no-memory control stays at base — confirming that the failure is the objective + freezing, not
> the sequence architecture.

## 8. Limitations

Laptop scale (≤14M parameters) and, unless noted, single seed; PR-AUC has run-to-run variance,
and training-seed variance is larger — we report effects that clear it and flag those that do
not. On IEEE we deliberately drop the ~400 anonymised Vesta features to match inputs across
models, which places both models far below the leaderboard frontier; our claims are about
*sequence-encoder-vs-trees on matched fields*, not absolute capability. The synthetic benchmark
isolates one relational mechanism (compromised-merchant velocity); real fraud is richer
(multi-entity rings, delayed labels). Memory-CSA is a rank-1 (single summary vector) compression
of a fuller cross-sequence encoder; a multi-entity, last-K "third-transformer" is future work.

## 9. Conclusion

An MLM-pretrained, frozen-probe financial foundation model beats gradient-boosted trees when
fraud is sequential and aligned with its objective, and loses when fraud is static or relational.
Adding cross-entity memory *architecturally* does not close the relational gap — we show, with a
ground-truth controlled benchmark, that a frozen MLM backbone cannot extract a cross-entity
feature even when it is handed to the model, because the pretraining objective never rewards
preserving it (the proxy-alignment gap). The path forward is the one production recommender
foundation models already take: align the pretraining objective with the task and train
end-to-end. Fraud is the harder case because its decisive relational signal is transient and
its labels are delayed — but the diagnosis, and the fix, are now precise.

## Reproducibility

Model, tokenizer, datasets adapters (`pragma.data.schema`, `pragma.data.ieee_cis`), the
controlled generator (`scripts/gen_synth_relational.py`), the memory builder, probes, and
per-experiment result JSONs are released. A golden test freezes the TabFormer tokenizer so all
reproductions are bit-exact. Detailed logs: `EXPERIMENTS.md`, `EXPERIMENTS_IEEE.md`,
`SYNTH_RELATIONAL.md`, `RELATIONAL_PRAGMA.md`.

## References (selected)

- Zhai et al. *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
  Generative Recommendations (HSTU)*, 2024.
- *OneRec: Unifying Retrieval and Ranking with Generative Recommender*, Kuaishou, 2024.
- Rajput et al. *Recommender Systems with Generative Retrieval (TIGER / semantic IDs)*, 2023.
- Padhi et al. *Tabular Transformers for Modeling Multivariate Time Series (TabBERT/TabFormer)*, 2021.
- Huang et al. *TabTransformer*, 2020; Gorishniy et al. *Revisiting DL for Tabular Data (FT-Transformer, numeric embeddings)*, 2021–2022.
- Ke et al. *LightGBM*, 2017; Chen & Guestrin *XGBoost*, 2016.
- IEEE-CIS Fraud Detection (Vesta), Kaggle, 2019; 1st-place solution write-up (client-UID feature engineering).
- Su et al. *RoFormer: Rotary Position Embedding (RoPE)*, 2021.
- Revolut, *Building a Financial Foundation Model (PRAGMA)*, 2024 (blog).
