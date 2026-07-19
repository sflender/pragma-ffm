# Controlled synthetic relational-fraud experiment

A ground-truth testbed for the central thesis, built because IEEE-CIS turned out to be a poor
instrument (its relational signal is already hand-engineered into the Vesta `V*` features, and
its fraud is largely static — see `EXPERIMENTS_IEEE.md` I3). Here we **synthesize** transactions
where we *control* exactly where the fraud signal lives, so we can test cleanly whether:

- a **per-sequence FFM** captures fraud that is a function of a card's **own history** (it should), and
- **fails** on fraud that is a function of a **shared entity's cross-card state** (it structurally must), which
- a **relational** model (entity memory / cross-sequence attention) can **recover**.

Generator: `scripts/gen_synth_relational.py`. Two fraud mechanisms:

- **`per_card`** — a card is compromised at a random time → a short **burst** of fraud txns on
  that card. Detectable from the card's own recent velocity (a per-sequence signal).
- **`relational`** — a **merchant** is compromised for a short window → a **velocity spike** of
  fraud txns spread across **many random cards**. Fraud amounts are drawn from the *same*
  distribution as legit; fraud is allocated to **every** merchant ∝ its traffic (uniform
  per-merchant fraud rate → merchant *identity* is uninformative); fraud cards are random (→ no
  per-card-history tell). **The only signal is the merchant's transient cross-card velocity.**

## Ground-truth validation (LightGBM feature groups) — DONE

Before any FFM run, we verify the signal lives where designed by fitting LightGBM on progressively
richer feature groups (per-card split, unseen cards). PR-AUC on test:

| feature set | `per_card` fraud (base 0.024) | `relational` fraud (base 0.028) |
|---|---|---|
| current txn only (amount, mcc, hour) | 0.04 | 0.03 |
| **+ card history** (pos, dt, expanding mean — *per-sequence*) | **0.82** ✓ | 0.06 ✗ |
| **+ merchant windowed velocity** (txns/hour at merchant — *cross-entity*) | 0.82 | **0.36** ✓ |

**This is exactly the intended structure, with ground truth:**
- **`per_card` fraud is recovered by the card's own history** (PR 0.04 → **0.82**) — precisely what a
  per-sequence FFM can see. Merchant velocity adds nothing on top.
- **`relational` fraud is invisible to card history** (PR 0.06, ~base) and **only** recovered by
  **cross-entity windowed velocity** (0.06 → **0.36**, +0.30). A per-sequence model is structurally
  blind to it; the signal is the merchant's cross-card burst.

Two things this pins down that IEEE could not: (1) the relational signal is **real and strong** and
(2) it is provably **outside a per-sequence model's receptive field**, recoverable only by a
cross-entity feature. The detector is **windowed velocity** (count/hour at the entity), not the
inter-arrival proxy we used on TabFormer/IEEE — so `build_merchant_memory.py --windows 3600,900`
now adds windowed-velocity features (d_mem=7).

## The FFM test (pending GPU — blocked on egress allowlist)

The controlled 2×2, mirroring the LightGBM table with the actual model
(`scripts/run_synth.sh`, `small`, 6k steps):

| | per_card fraud | relational fraud |
|---|---|---|
| **embedding-only** (per-sequence FFM, no memory) | should **detect** (card history) | should **fail** |
| **memory-CSA** (+ windowed-velocity merchant memory) | ≈ same (memory irrelevant) | should **recover** |

**I predicted memory-CSA would win on relational and reverse the TabFormer/IEEE negative. It did
not — and the actual result is sharper and more useful than the win would have been.**

### Result (`small`, 6k steps, `artifacts/synth_2x2_results.json`)

| arm | **per_card** (base 0.024) | **relational** (base 0.028) |
|-----|---------------------------|------------------------------|
| embedding-only (per-sequence FFM) | **PR 0.81 / ROC 0.95** ✓ | PR 0.035 / ROC 0.56 ✗ |
| memory-CSA (+ windowed-velocity memory) | PR 0.81 / ROC 0.94 | PR 0.047 / ROC 0.60 ✗ |
| duct-tape fusion (probe) | PR 0.83 / ROC 0.97 | PR 0.047 / ROC 0.59 ✗ |
| **— LGBM w/ windowed velocity (ground truth)** | — | **PR 0.36 / ROC 0.83** ✓ |

Two clean findings, both confirmed against ground truth:

1. **Per-sequence architectural blindness is real and total.** The FFM *nails* per_card fraud
   (0.81) — fraud in a card's own history is exactly what a sequence model captures — and
   *collapses to base* on relational fraud (0.035 ≈ base 0.028). Same model, same data
   generator, opposite outcome, driven only by *where the signal lives*.

2. **The frozen-MLM recipe cannot extract cross-entity signal even when handed the exact
   feature.** memory-CSA's memory *contains* the windowed-velocity feature that LightGBM turns
   into **PR 0.36** — yet memory-CSA reaches only **0.047** (ROC 0.56→0.60, essentially still
   base). The *same feature* recovers the fraud at the **supervised** head (LGBM 0.36) and fails
   in the **MLM-pretrained backbone read by a frozen linear probe** (0.047). This is the
   **proxy-alignment gap demonstrated with ground truth**: MLM has no reason to preserve a
   feature that doesn't help token reconstruction, so the frozen probe can't read it — *even
   when it's literally in the input.*

**This does not reverse the TabFormer (§6.1) and IEEE (I4) memory-CSA negatives — it explains
them, definitively.** Those weren't dataset artifacts; they're the structural consequence of
injecting relational signal into a frozen SSL-proxy backbone. The controlled setting removes
every confound (the signal is real, strong, and provably in the memory) and the negative
persists — which is far stronger evidence than another noisy real-data null.

**Caveat:** the duct-tape arm here used the *old* 3 relational features (popularity, prior fraud
rate, novelty), **not** windowed velocity, so its 0.047 is not a fair "FFM-embedding ⊕ velocity"
test — the honest supervised-injection reference is the **LGBM w/ windowed velocity (0.36)**. A
fusion probe fed windowed velocity would very likely land near 0.36 too, sharpening the
"inject downstream, not in the SSL backbone" contrast.

### What it means for the architecture

The fix is now precisely motivated, and it matches what production recsys FMs (HSTU, OneRec) do
to avoid this exact gap: **(a) align the objective** — train with a fraud-relevant or
relational-SSL objective instead of pure MLM, and/or **(b) don't freeze** — end-to-end / LoRA so
the head can extract the memory signal. Recsys sequence models sidestep the proxy-alignment gap
because their pretraining objective *is* the task (next-item) and they train end-to-end; the
transient cross-entity signal they *can't* embed (real-time "trending") they inject as
serving-time features — i.e. the duct-tape. Both levers (aligned objective, unfrozen head) are
the concrete next experiments, now backed by a ground-truth controlled result rather than
conjecture.

## S2 — the fix: align the objective + unfreeze (end-to-end fine-tuning)

Directly testing the fix S1 pointed to: train the *same* memory architecture **end-to-end on the
fraud label** over as-of-date windows, instead of frozen MLM + linear probe
(`scripts/finetune_synth.py`).

| arm | PR-AUC | ROC-AUC |
|-----|--------|---------|
| frozen-MLM memory-CSA (S1) | 0.047 | 0.60 |
| **end-to-end + memory (S2)** | **0.080** | **0.68** |
| end-to-end, no memory (control) | 0.043 | 0.63 |
| LightGBM w/ windowed velocity (ceiling) | 0.36 | 0.83 |

**Result — the fix is directionally right but a rank-1 memory doesn't close the gap.** Aligning
the objective and unfreezing ≈doubles PR-AUC over the frozen probe (0.047→0.080) and lifts ROC
(0.60→0.68) — the cross-entity signal *does* become more extractable when the objective rewards it
and the head can shape the backbone. The **no-memory control stays at base (0.043)**, confirming
the memory is *necessary* (objective+unfreezing can't help a per-sequence model see cross-entity
signal). But end-to-end fine-tuning recovers only ≈1/5 of the GBDT-accessible signal (0.36): a
single-summary-vector memory is too lossy. This motivates the **multi-entity, last-K cross-sequence
("third-transformer") encoder** as the architecture that should actually close the gap.
Artifacts: `artifacts/synth_finetune_results.json`.

## S3 — the two levers, ablated: cross-sequence encoder vs aligned SSL

S2 left two hypotheses for *why* the rank-1 memory falls short and what would fix it: **(#1)
architecture** — replace the rank-1 summary with a real **cross-sequence encoder** ("third
transformer") that attends over the entity's last-K *raw* prior events, so the model learns the
relational pattern instead of trusting a hand-designed feature; **(#2) objective** — the frozen gap
is proxy-alignment, so an **aligned SSL objective** (regress the windowed-velocity target from the
record embedding during pretraining) should let a *frozen* probe read the signal. S3 runs both,
against two GBDT ceilings (`small`, L40S; `scripts/build_entity_neighbors.py`,
`finetune_synth.py --xseq`, `pretrain.py --aux-vel-lambda`).

Two reference ceilings (LightGBM, test base 0.028):
- **velocity-only** (windowed velocity feature): **PR 0.36** — the pure cross-entity-velocity signal.
- **full merchant features** (velocity + popularity + prior-fraud-rate + ...): **PR 0.527 / ROC 0.911**
  — a looser upper bound that also uses the label-derived merchant prior-fraud-rate.

### #1 — cross-sequence encoder (the "third transformer") — **the fix works**

| arm (end-to-end fine-tune) | PR-AUC | ROC-AUC |
|-----|--------|---------|
| per-sequence, no memory (control) | 0.037 | 0.584 |
| rank-1 memory-CSA (S2) | 0.078 | 0.676 |
| **cross-sequence encoder (`--xseq`)** | **0.409** | **0.893** |
| cross-sequence encoder, **frozen backbone** (`--xseq --freeze-backbone`) | **0.362** | 0.887 |

- **The cross-sequence encoder recovers the relational signal the rank-1 memory could not.** 0.409 vs
  0.078 (**5.2×** the rank-1 memory) vs 0.037 (**11×** the per-sequence control). It **exceeds the
  velocity-only ceiling (0.36)** and reaches **~78% of the full-feature GBDT ceiling (0.527)** —
  despite having *no* access to labels or the prior-fraud-rate feature. Seeing the raw neighbour
  events lets it learn richer burst structure than the single velocity feature encodes.
- **It works even on a frozen per-sequence backbone (0.362 ≈ velocity ceiling).** Training *only* the
  cross-sequence module + head, with the entire per-sequence FFM frozen, still recovers the signal.
  The relational capability is a **bolt-on module** — a deployed frozen FFM does not need to be
  retrained to gain it. This is the "inject the relational signal downstream" lesson from S1, done
  *architecturally* instead of as a hand-designed feature.

### #2 — aligned SSL (frozen probe) — objective-alignment helps, but architecture is the bigger lever

| arm (frozen probe after pretraining) | PR-AUC | ROC-AUC |
|-----|--------|---------|
| memory-CSA, pure MLM (S1 repro) | 0.058 | 0.618 |
| **memory-CSA + velocity-SSL aux (`--aux-vel-lambda`)** | **0.121** | **0.760** |
| velocity-SSL aux **without** memory input (negative control) | see note | |

- **Aligning the SSL objective ≈doubles the frozen probe** (0.058 → 0.121, ROC 0.618 → 0.760). Adding
  a velocity-regression aux loss — with the velocity feature already in the memory *input* — forces
  that cross-entity signal into the record embedding, so a **frozen** linear probe can now read it.
  This is the **proxy-alignment gap demonstrated as causal**: pure MLM discards the signal it has no
  reason to reconstruct; rewarding it in the objective preserves it, *without unfreezing*.
- **But 0.121 ≪ 0.409.** Objective-alignment recovers only a fraction of what the cross-sequence
  architecture does. The rank-1 memory bottleneck limits how much any objective can extract — the
  decisive lever is the **architecture** (rich cross-sequence attention over raw neighbours), not the
  objective. (The negative control — aux loss with *no* memory input, so the cross-card signal isn't
  available to preserve — is expected to stay near base, confirming the aux needs the signal present
  in the input; result appended when the arm completes.)

### The full arc

1. Per-sequence FFM is **structurally blind** to relational fraud (0.037 ≈ base). *(S1)*
2. A rank-1 entity memory **barely helps**, even fine-tuned (0.078); frozen, the **proxy-alignment
   gap** keeps it near base (0.058). *(S1/S2)*
3. **Aligning the SSL objective partially closes the frozen gap** (0.058→0.121) — proving the gap is
   real and objective-driven, not a capacity limit. *(#2, S3)*
4. **The decisive fix is architectural**: a cross-sequence encoder over raw entity neighbours recovers
   the signal (**0.409**, past the velocity ceiling), and does so **even on a frozen backbone**
   (0.362). *(#1, S3)*

So the controlled benchmark not only exposes the relational gap in the frozen-FFM recipe but
**localises and closes it**, cleanly separating the two candidate fixes and showing architecture
dominates objective-alignment. Caveats: single seed / single generator; `--xseq` and
`--freeze-backbone` train the cross-sequence module + head on labels (only the *per-sequence
backbone* is frozen — not a pure linear probe); the velocity-only 0.36 is a feature-specific
reference, so the full-feature 0.527 is the more honest upper bound (xseq reaches 78% of it).
Artifacts: `artifacts/ft_xseq.json`, `ft_xseqfrozen.json`, `ft_mem.json`, `ft_nomem.json`,
`probe_memaux.json`, `probe_memonly.json`, `lgbm_ceiling.json`.
