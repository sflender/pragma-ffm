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

**Prediction:** memory-CSA **wins on relational fraud and ties on per_card** — which would
**reverse** the memory-CSA negative from TabFormer (§6.1) and IEEE (I4). The reason those were
negative is now precise: there the relational signal was weak/absent or the memory lacked the
right feature; here it is real *and* the memory captures it. This is the controlled proof that the
memory mechanism recovers cross-entity signal a per-sequence model cannot — the positive result the
thesis needs before investing in the full multi-entity cross-sequence ("third-transformer")
architecture on real relational data (IBM AML / Elliptic).

**Status:** generator + memory + pipeline validated locally end-to-end; the GPU FFM run is queued
and blocked only on the network egress allowlist reset (needs `api.runpod.io`, `ntfy.sh` re-added).
Data is generated on-pod (no external dependency once RunPod is reachable).
