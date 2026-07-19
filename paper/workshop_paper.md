# When Do Financial Foundation Models Beat Gradient-Boosted Trees? Sequential vs. Relational Fraud, the Proxy-Alignment Gap, and a Cross-Sequence Fix

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
pretraining objective *is* the task and they train end-to-end. Aligning the objective and
unfreezing the backbone *partially* recovers the signal (≈2× the frozen probe) but a rank-1
memory does not close the gap to the GBDT. Finally, we **close the gap architecturally**: a
**cross-sequence encoder** that attends over an entity's last-K *raw* prior events — rather than a
rank-1 precomputed summary — recovers the relational signal (PR-AUC **0.41**, past the 0.36
velocity ceiling and ~78% of the full-feature GBDT), and does so **even on a frozen per-sequence
backbone** (0.36), making relational capability a *bolt-on* module. Ablating the two candidate
fixes on ground truth, **architecture dominates objective-alignment** (0.41 vs 0.12): the training
recipe helps, but a rich cross-sequence encoder is what actually closes the gap. On real graph
fraud (Elliptic) the advantage is **bounded** — the encoder does not beat rich hand-engineered
neighbour aggregates and adds little on top — locating its value where the available neighbour
summary is *lossy* and the relational signal *transient*.

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

4. **Closing the gap: cross-sequence encoding beats objective-alignment (§8).** On the same
   controlled benchmark we ablate the two candidate fixes. A **cross-sequence encoder** — the
   target event cross-attends over its entity's last-K *raw* prior events, learned end-to-end —
   recovers the relational signal (PR-AUC 0.41, past the velocity ceiling), and remarkably keeps
   most of it with the per-sequence backbone **frozen** (0.36); relational capability is a bolt-on
   module. An aligned self-supervised objective (velocity regression) closes only part of the
   frozen gap (0.06→0.12) and only when the signal is present in the input. Architecture is the
   decisive lever.

All code, the synthetic generator, the cross-sequence encoder, and per-experiment result files
are released.

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
**controlled synthetic generator** (§6) with tunable fraud structure; (iv) **Elliptic** (§8) —
203K Bitcoin transaction nodes / 234K payment edges with real AML labels, for graph external validity.

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

**S2: aligning the objective and unfreezing helps — but a rank-1 memory does not close the gap.**
We fine-tune the *same* architecture **end-to-end on the fraud label** over as-of-date windows on
the synthetic `relational` data:

| arm | PR-AUC | ROC-AUC |
|---|---|---|
| frozen-MLM memory-CSA (S1) | 0.047 | 0.60 |
| **end-to-end + memory (S2)** | **0.080** | **0.68** |
| end-to-end, no memory (control) | 0.043 | 0.63 |
| LightGBM w/ windowed velocity (ceiling) | 0.36 | 0.83 |

Two confirmations and a caveat. Aligning the objective and unfreezing **nearly doubles** PR-AUC
over the frozen-MLM memory-CSA (0.047→0.080) and lifts ROC (0.60→0.68): the fix is *directionally*
correct — the cross-entity signal becomes more extractable once the objective rewards it and the
head can shape the backbone. The **no-memory control stays at base** (0.043), confirming the
memory is *necessary*: objective-alignment and unfreezing alone cannot make a per-sequence model
see cross-entity signal. But end-to-end fine-tuning of this **rank-1** (single-summary-vector)
memory recovers only ≈1/5 of the GBDT-accessible signal (0.36). The training recipe was the wrong
first suspect *and* an incomplete fix: closing the gap needs a richer **multi-entity, last-K
cross-sequence** encoder (§8), not just an aligned objective over a compressed memory. The
direction is confirmed; the magnitude says the *architecture* still matters.

## 8. Closing the gap: cross-sequence encoding vs. aligned objective

S2 leaves two candidate fixes for the rank-1 memory's shortfall. We ablate both on the synthetic
`relational` data (`small`), against two GBDT ceilings: **velocity-only** (the single cross-entity
feature, PR-AUC 0.36) and **full merchant features** (velocity + popularity + label-derived prior
fraud rate, **0.53**) — the latter a looser but more honest upper bound.

**(1) A cross-sequence encoder — the "third transformer."** Instead of a rank-1 precomputed
summary, the target event cross-attends over its entity's **last-K raw prior events** (across all
cards, strictly as-of-date), each encoded by the shared event encoder, with a learned recency
signal and a shallow transformer over the K neighbours. The model *learns* the relational pattern
from the concurrent cross-card activity rather than reading a hand-designed feature.

**(2) An aligned self-supervised objective.** During MLM pretraining, an auxiliary head regresses
the (data-derived) windowed-velocity target — already present in the memory *input* — from the
record embedding, forcing the cross-entity signal into the frozen representation. The backbone is
then read by the usual frozen linear probe.

| arm | PR-AUC | ROC-AUC |
|---|---|---|
| per-sequence FFM, no memory (control) | 0.04 | 0.58 |
| rank-1 memory-CSA, end-to-end (S2) | 0.08 | 0.68 |
| memory-CSA, pure MLM, frozen probe | 0.06 | 0.62 |
| **+ velocity-SSL aux, frozen probe** (aligned objective) | **0.12** | 0.76 |
| velocity-SSL aux, *no memory input* (negative control) | 0.04 | 0.58 |
| **cross-sequence encoder, end-to-end** | **0.41** | 0.89 |
| **cross-sequence encoder, frozen backbone** | **0.36** | 0.89 |
| — LightGBM, velocity-only / full features (ceilings) | 0.36 / **0.53** | 0.83 / 0.91 |

**The fix is architectural.** The cross-sequence encoder reaches **0.41** — **5× the rank-1
memory** and **11× the per-sequence control** — *exceeding* the velocity-only ceiling and reaching
~78% of the full-feature GBDT, despite having no access to labels or the prior-fraud-rate feature:
seeing the raw neighbour events lets it learn richer burst structure than any single velocity
statistic encodes. Strikingly, with the **entire per-sequence backbone frozen** and only the
cross-sequence module + head trained, it still recovers **0.36** — the relational capability is a
**bolt-on**: a deployed frozen FFM gains it from an add-on module, no re-pretraining.

**Objective-alignment helps, but is the smaller lever.** Adding the velocity-SSL aux ≈doubles the
frozen probe (0.06→0.12, ROC 0.62→0.76): a *causal* demonstration of the proxy-alignment gap —
pure MLM discards the signal it has no reason to reconstruct; rewarding it in the objective
preserves it, without unfreezing. But 0.12 ≪ 0.41, and the negative control (aux with *no* memory
input) stays at base (0.04) — so aligning the objective needs *both* the signal present in the
input and a rich enough pathway; on a rank-1 memory it cannot match the architecture.

**Full arc.** Per-sequence blindness (0.04) → rank-1 memory barely helps / frozen proxy-alignment
gap (0.06) → aligned objective partially closes it (0.12) → cross-sequence architecture recovers
the signal (0.41), even frozen-backbone (0.36). The controlled benchmark does not just *expose*
the relational gap in the frozen-FFM recipe — it **localises and closes it**, and separates the
two fixes cleanly enough to show architecture dominates objective-alignment.

**External validity (Elliptic): the advantage is bounded.** We port the same cross-sequence
encoder to the **Elliptic Bitcoin** dataset (203K transaction nodes, 234K payment edges,
illicit-vs-licit AML labels; standard temporal split), where "cross-sequence" becomes cross-node
graph attention and the 165 features split into 93 *local* and 72 *aggregated* (engineered
one-hop neighbour) features — a natural map to blind / rank-1-summary / raw-neighbour. Holding a
shared MLP encoder fixed, attention over raw neighbours (`+xseq`, illicit PR-AUC ≈0.59–0.63)
**beats no-neighbours (≈0.55–0.58) and naïve mean-pooling but does *not* beat the engineered
aggregates (≈0.68–0.69), and adds essentially nothing on top of them** (`+agg+xseq` ≈ `+agg`). The
mechanism still extracts relational signal — it is simply *redundant* here. This does not overturn
the synthetic result; it **bounds** it: the cross-sequence encoder's advantage scales with how
**lossy** the available neighbour summary is (rank-1 velocity vs. 72 multi-statistic features) and
how **transient** the signal is (a compromise burst vs. static laundering structure). Where a rich
engineered summary already exists and the signal is stationary — and, consistent with §4, where
GBDTs dominate the absolute numbers (LightGBM 0.79–0.82 vs. the MLP family) — raw-neighbour
attention is not a free win. (Single-seed, weak-base, K≤16 sampled neighbours; a matched-receptive-field,
multi-seed study is needed for a hard real-data verdict. Details: `ELLIPTIC.md`.)

## 9. Limitations

Laptop scale (≤14M parameters) and, unless noted, single seed; PR-AUC has run-to-run variance,
and training-seed variance is larger — we report effects that clear it and flag those that do
not. On IEEE we deliberately drop the ~400 anonymised Vesta features to match inputs across
models, which places both models far below the leaderboard frontier; our claims are about
*sequence-encoder-vs-trees on matched fields*, not absolute capability. The synthetic benchmark
isolates one relational mechanism (compromised-merchant velocity); real fraud is richer
(multi-entity rings, delayed labels). Our one real-relational test (Elliptic, §8) is a first pass:
single-seed with large run-to-run variance, a weak shared MLP base, and K≤16 sampled neighbours
with local-only features — enough to bound the claim (raw attention did not beat rich engineered
aggregates) but not to settle whether a matched-receptive-field, multi-seed study would. The
cross-sequence encoder and its `--freeze-backbone`
variant train the cross-sequence module and head on labels — "frozen" refers to the *per-sequence
backbone*, not a pure linear probe; and it is single-entity (merchant), where the synthetic signal
lives. Finally, the cross-sequence encoder exceeds the *velocity-only* ceiling because it accesses
raw neighbour events (richer than one hand-designed feature), so 0.36 is a feature-specific
reference and the full-feature 0.53 is the honest upper bound.

## 10. Conclusion

An MLM-pretrained, frozen-probe financial foundation model beats gradient-boosted trees when
fraud is sequential and aligned with its objective, and loses when fraud is static or relational.
Adding cross-entity memory as a *rank-1 summary* does not close the relational gap — we show, with
a ground-truth controlled benchmark, that a frozen MLM backbone cannot extract a cross-entity
feature even when it is handed to the model, because the pretraining objective never rewards
preserving it (the proxy-alignment gap). We then close the gap: a **cross-sequence encoder** that
attends over an entity's last-K raw prior events recovers the relational signal (0.41, past the
velocity ceiling), and keeps most of it as a **bolt-on to a frozen backbone** (0.36); ablating the
two candidate fixes, this architecture dominates objective-alignment (0.41 vs 0.12). The path
forward for relational fraud is thus not merely the recipe production recommenders use (aligned
objective, end-to-end) but an **architecture that reads concurrent cross-sequence activity** — and
it can be added to an existing frozen FFM without re-pretraining. Fraud remains the harder case
because its decisive relational signal is transient and its labels are delayed, and real-data and
multi-entity validation are the next steps — but the diagnosis, and the fix, are now precise and
demonstrated.

## Reproducibility

Model, tokenizer, datasets adapters (`pragma.data.schema`, `pragma.data.ieee_cis`), the
controlled generator (`scripts/gen_synth_relational.py`), the memory builder, the cross-sequence
neighbour builder (`scripts/build_entity_neighbors.py`) and encoder (`--xseq` /
`--freeze-backbone`), the aligned-SSL aux head (`pretrain.py --aux-vel-lambda`), probes, and
per-experiment result JSONs are released. A golden test freezes the TabFormer tokenizer so all
reproductions are bit-exact. The Elliptic adapter (`scripts/build_elliptic.py`) and node arms
(`scripts/elliptic_relational.py`) reproduce the external-validity test. Detailed logs:
`EXPERIMENTS.md`, `EXPERIMENTS_IEEE.md`, `SYNTH_RELATIONAL.md`, `ELLIPTIC.md`, `RELATIONAL_PRAGMA.md`.

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
