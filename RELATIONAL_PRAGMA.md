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

### B. Entity-memory cross-attention *(the recommended first *model*)*
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

See `scripts`-style experiment `relational_probe.py`: causal merchant-level features on the
IBM TabFormer test subsample.

> **Result placeholder** — filled from the run:
> - standalone merchant features' ROC/PR-AUC,
> - within-sequence LightGBM PR-AUC vs +relational (the lift).

If even simple merchant aggregates lift PR-AUC over a within-sequence model, the relational
direction is validated cheaply — before building B/C.

## 7. Data caveat (important)

IBM TabFormer fraud is **rule-injected and largely per-user (synthetic)**, so its
*relational* signal is likely **weaker than real fraud**. Any lift measured here therefore
**understates** the real-world upside of relational modelling. A faithful test needs a
dataset with genuine cross-entity fraud (fraud rings, compromised merchants) — e.g., the
kind of proprietary corpus PRAGMA was actually trained on.

## 8. Roadmap

1. **(A)** Relational features → probe/baseline; measure lift. *(this repo can start today)*
2. **(B)** Entity-memory cross-attention; pretrain with MLM + masked-entity-context; probe.
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
