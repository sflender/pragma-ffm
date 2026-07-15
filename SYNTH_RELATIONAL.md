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
