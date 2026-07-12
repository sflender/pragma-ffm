# Relational PRAGMA — research outline

*A proposal to extend the per-sequence financial foundation model with cross-entity
(relational) structure. Motivated by an ablation in this repo: doubling the within-sequence
context window (L=128→256) did **not** improve fraud PR-AUC, because the useful signal is
local — the model is starved of **relational** signal, not of sequence length.*

## 1. Motivation — the per-sequence blind spot

`mini-PRAGMA` (and PRAGMA itself) encode **one `(user, card)` sequence at a time**. The
History Encoder attends only *within* a card's own event stream. This makes the model
structurally blind to the patterns that define much of real-world fraud, which are
**cross-entity**:

- **Compromised merchant / terminal** — many *different* cards get hit at the same merchant
  in a short window. "This merchant suddenly has anomalous activity across 50 cards" is
  invisible to a model that sees each card alone.
- **Card-testing / enumeration rings** — small charges fanned across merchants, or one
  merchant probed by many stolen cards.
- **Shared-entity contagion** — `merchant`, `MCC`, `city`, `zip` are shared across
  sequences; a transaction at an entity that is "hot" right now is riskier.

Worse, the current tokenizer **crc32-hashes `merchant_name` into 4,096 buckets**, so entity
*identity* is partially destroyed before the model even starts — and the architecture never
aggregates across cards that share an entity.

**Evidence from this repo.** Extending context to L=256 (89.5% of fraud targets have ≥256
*real* prior events, so it's genuine history, not padding) left ROC-AUC flat and *hurt* the
high-precision tail — the classic signature of "more context = more noise, not more signal."
Meanwhile a quick causal **merchant-level** feature carries independent fraud signal (see
`§6`). The lever is relational structure, not sequence length.

## 2. Goal

Learn representations of banking events that incorporate **the recent behaviour of the
entities each event touches** (merchant, MCC, geography, and ultimately counterparties),
while keeping PRAGMA's core recipe: self-supervised pretraining + a frozen backbone read
out by a linear probe. Adapt to fraud (and other tasks) with no fine-tuning.

## 3. Design space (cheap → ambitious)

### A. Relational features into the probe/baseline *(hours; strong baseline)*
Causal, entity-level aggregates as side-inputs: merchant rolling txn/decline/fraud rate,
# distinct cards at a merchant in the last hour/day, velocity at `(merchant, MCC, zip)`,
card–merchant novelty. Concatenate to the frozen record embedding before the logistic head.
This quantifies the ceiling of hand-built relational signal and is the honest baseline any
learned approach must beat.

### B. Entity-memory cross-attention *(implemented + tested — surprisingly **lost** to A; see §6.1)*
Keep the event/history encoders, but give each event access to a **merchant memory**: a
learned embedding summarising that merchant's *recent global* activity across **all** cards,
as of the event's timestamp. Concretely:

- Maintain a streaming, time-decayed memory vector `m_e(t)` per entity `e` (merchant, and
  optionally MCC/zip), updated from every transaction touching `e` (à la TGN memory).
- In the Event Encoder, add **relational tokens**: the current `m_merchant(t)`,
  `m_mcc(t)`, `m_zip(t)` for this event, so the `[EVT]` vector fuses "what is this card
  doing" with "what is happening at this merchant right now."
- Train end-to-end; memory read is **as-of-date** (past-only) so it stays leakage-safe and
  matches the sliding-window eval.

This adds relational context with **no full graph** — O(events), streaming, scalable — and
slots into the existing tokenizer/encoder with modest changes.

### C. Temporal heterogeneous GNN *(the "proper", heaviest version)*
Build the card↔merchant (and ↔MCC/geo) **temporal bipartite graph**; run a temporal GNN
(TGN / TGAT) so signal propagates across cards sharing entities. Node embeddings feed the
downstream probe. Highest ceiling, highest engineering cost, and the standard at large
payment networks. Best treated as the north-star after B proves the relational lift.

## 4. Pretraining objectives

Keep **MLM** (reconstruct masked `(field,value)` tokens) and add relational self-supervision:

- **Masked-entity-context**: mask the relational tokens and reconstruct entity-level
  summaries (e.g., merchant's next-hour txn count bucket) — forces the memory to be
  predictive.
- **Contrastive co-occurrence**: events sharing an entity within a short window are positives;
  random events are negatives — pulls together representations that are relationally linked.
- **Next-entity / link prediction** (GNN variant): predict the next merchant a card visits.

All objectives are **causal / as-of-date** to preserve the leakage-safe framing.

## 5. Evaluation

- Primary: fraud **PR-AUC** on the same stratified sliding-window as-of-date subsample used
  here (n=152,928, base 0.0191), so numbers are directly comparable to the per-sequence FFM
  (PR-AUC **0.807**) and the blog (0.786).
- Ablations: per-sequence FFM vs +relational-features (A) vs +entity-memory (B) vs GNN (C);
  and *which* entities matter (merchant vs MCC vs geo).
- Stress test: recall@high-precision (where the per-sequence model degraded) — the region
  relational signal should most help.

## 6. Empirical seed (this repo)

Causal merchant-level features on the IBM TabFormer test subsample (n=152,928, base 0.0191),
LightGBM fit on the matched train subsample (same regime as the FFM probe).

**Standalone signal (test subsample):**

| feature | ROC-AUC | PR-AUC | note |
|---|---|---|---|
| `m_prior_fraud_rate` (merchant's past fraud rate) | 0.847 | 0.314 | **strong** — but *uses past labels* |
| `m_pop` (merchant popularity, log prior txns) | 0.373 | 0.014 | label-free, weak (inverted: popular ⇒ less fraud) |
| `cm_new` (first time this card @ this merchant) | 0.536 | 0.029 | label-free, weak |

**Complementarity (LightGBM, same fit regime):**

| features | PR-AUC | ROC-AUC |
|---|---|---|
| within-sequence only | 0.881 | 0.991 |
| relational only | 0.356 | 0.923 |
| **within-sequence + relational** | **0.920** | 0.994 |

→ **Relational features add +0.038 PR-AUC** on top of a strong within-sequence model — signal
a per-sequence encoder structurally cannot access. **Validated.**

**Honest caveats.**
1. The lift is driven by `m_prior_fraud_rate`, which is **label-based** (past fraud at the
   merchant). It's causal, but in production fraud labels arrive with weeks of delay, so
   real-time availability is weaker; a faithful version needs label-delay simulation.
2. The **label-free** relational features (velocity/novelty) are weak *here* — consistent
   with TabFormer's fraud being rule-injected **per-user**, so its label-free relational
   structure is thin. Real fraud (rings, compromised merchants) carries strong label-free
   relational signal, so this **understates** the real-world upside (see §7).
3. These LightGBM numbers are fit on the balanced train subsample (to match the FFM probe's
   regime), so they run higher than the blog's full-data-fit LightGBM (0.369); the valid
   takeaway is the **+0.038 relative lift**, not the absolute values.

## 6.1. Approach B tested on GPU: entity-memory cross-attention *loses to the duct-tape*

We implemented **Approach B** (§3B) as a `MemoryCrossAttention` module: a per-event causal
5-dim merchant memory (`merchant_mem.npz`) is projected to `d_model` and cross-attended by the
History Encoder output, trained end-to-end under the **same MLM objective**, then read out by
the **same frozen linear probe**. Head-to-head against the duct-tape fusion (§3A) on one
controlled GPU run — identical data / split / tokenizer / **8k steps** / batch / LR — so the
only variable is *where the relational signal enters*:

| arm | where relational signal enters | probe head | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| embedding-only (no-mem backbone) | — | linear | 0.747 | 0.980 |
| **memory-CSA** (Approach B) | **backbone** (MLM-pretrained) | linear | **0.694** | 0.975 |
| duct-tape fusion (Approach A) | **probe** (supervised) | logreg | 0.786 | 0.984 |
| duct-tape fusion (Approach A) | **probe** (supervised) | LightGBM | **0.844** | 0.988 |

→ **The architectural memory solution not only lost to the duct-tape — it fell *below* the
plain no-mem baseline (−0.053 PR-AUC).** The duct-tape concat beat it by +0.09 (logreg) to
+0.15 (LightGBM). Injecting relational signal into the backbone *hurt*.

**Why (the actual finding).** *Where* you inject relational signal matters more than how
elegant the injection is:
1. **Objective mismatch.** The backbone trains on **MLM** (self-supervised token
   reconstruction). The strongest relational feature — the merchant's prior fraud rate — is
   **label-derived** and useless for reconstructing masked tokens, so MLM gives the backbone
   **no gradient signal** to preserve it. The cross-attention pathway optimises for
   reconstruction, not fraud. The duct-tape injects the same features at the **supervised**
   probe, where they directly serve the fraud objective.
2. **Frozen-backbone bottleneck.** Any captured signal must survive as a *linearly-readable
   direction* in the frozen last-position embedding — lossy versus handing the raw feature to
   the probe.
3. **Cost without benefit.** The mem pathway added a module (+38k params, a new route) on the
   *same* step budget, mildly perturbing the representation with nothing to compensate.

**Takeaway that reframes this direction:** self-supervised pretraining **cannot see the
fraud label**, so relational signal that only pays off downstream should be injected
**downstream** (Approach A), not baked architecturally into the SSL backbone. This answers
§9's open question ("does relational signal help the linear-probe readout, or need
fine-tuning?") for the *pretraining-injection* route: under a frozen backbone + MLM, no.

**Caveats — not yet a settled negative.** This is 8k steps with the mem pathway frozen at
probe time. Untested rescues: (i) much longer pretraining; (ii) a **relational auxiliary
objective** during pretraining (§4 masked-entity-context / contrastive co-occurrence) so MLM
*does* reward using the memory; (iii) light fine-tuning of the mem pathway on the fraud task;
(iv) a dataset with real label-free relational fraud (§7), where the memory would carry
reconstruction-relevant structure. The clean result stands: **elegance ≠ signal placement.**

## 7. Data caveat (important)

IBM TabFormer fraud is **rule-injected and largely per-user (synthetic)**, so its
*relational* signal is likely **weaker than real fraud**. Any lift measured here therefore
**understates** the real-world upside of relational modelling. A faithful test needs a
dataset with genuine cross-entity fraud (fraud rings, compromised merchants) — e.g., the
kind of proprietary corpus PRAGMA was actually trained on.

## 8. Roadmap

1. **(A)** Relational features → probe/baseline; measure lift. ✅ *done — +0.04 (logreg) / +0.10 (LGBM) PR-AUC; §6, §6.1.*
2. **(B)** Entity-memory cross-attention; pretrain with MLM + masked-entity-context; probe.
   ✅ *implemented + tested (MLM-only) — **lost to (A)**, fell below baseline; §6.1. Next: add
   a relational SSL objective (§4) so MLM rewards the memory, then re-test.*
3. **(C)** Temporal GNN; compare ceiling.
4. Scale entities (counterparty graph), add relational SSL objectives, test on a dataset
   with real relational fraud.

## 9. Open questions

- Time-decay / memory horizon per entity (fast for card-testing, slow for merchant risk)?
- Cold-start entities (first-seen merchants) — fall back to MCC/geo priors?
- Leakage discipline: entity memory must be strictly as-of-date, and label-derived entity
  features (e.g., merchant prior fraud rate) need realistic label-delay simulation.
- Does relational signal help the *linear-probe* readout, or does it require light
  fine-tuning to surface?
