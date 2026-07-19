# Elliptic — external validity for the cross-sequence encoder on real relational fraud

The synthetic §8 result (`SYNTH_RELATIONAL.md` S3) showed a **cross-sequence encoder** — attending
over an entity's last-K *raw* neighbours — recovers relational fraud a per-sequence model and a
*rank-1* precomputed memory cannot. This tests whether that holds on **real** relational fraud, on
the **Elliptic Bitcoin dataset** (Weber et al., 2019): 203,769 transaction nodes, 234,355 payment
edges, illicit-vs-licit (AML) labels. Standard temporal split (steps 1–34 train / 35–49 test);
metric is the illicit-class PR-AUC / ROC-AUC / F1 the Elliptic literature uses.

**The mapping.** Elliptic is a *graph*, not per-entity event sequences, so "cross-sequence" becomes
"cross-node graph attention" — the same `CrossSequenceEncoder`, over graph neighbours. Its 165
features split into **93 local** (a node's own tx features) and **72 aggregated** (hand-engineered
one-hop neighbour summaries), which map onto our thesis:

| our synthetic arm | Elliptic analog |
|---|---|
| per-sequence FFM (blind) | MLP on **93 local** features |
| rank-1 precomputed memory | + **72 aggregated** neighbour features |
| cross-sequence encoder (raw) | + attention over **raw neighbour nodes'** local features |

All torch arms share one local-encoder MLP; the clean contrast is **+xseq vs +agg** (learned
attention over raw neighbours vs the engineered neighbour summary), plus **+agg+xseq** (does raw
attention add anything *on top of* the summary?). `scripts/build_elliptic.py`,
`scripts/elliptic_relational.py`.

## Result — the synthetic win does NOT replicate here

Illicit-class metrics on the temporal test split (base 6.5%). Two runs shown where available to
convey the (large) single-seed variance:

| arm | PR-AUC | illicit-F1 |
|---|---|---|
| LightGBM, local (reference) | 0.790 | 0.789 |
| LightGBM, local + aggregated (reference) | **0.816** | **0.826** |
| MLP local (blind) | 0.55 / 0.58 | 0.65 / 0.67 |
| MLP local **+ aggregated** (rank-1 analog) | **0.68 / 0.69** | 0.70 / 0.72 |
| MLP local + meanpool raw neighbours | 0.60 / 0.45 | 0.60 / 0.51 |
| MLP local **+ xseq** (raw-neighbour attention) | 0.63 / 0.59 | 0.62 / 0.62 |
| MLP local **+ aggregated + xseq** (both) | 0.70 | 0.68 |

**Three honest findings:**

1. **Raw-neighbour attention does not beat the engineered aggregates.** `+xseq` (0.59–0.63) sits
   *below* `+agg` (0.68–0.69) in both runs — the opposite of the synthetic result, where the
   cross-sequence encoder beat the (rank-1) summary 5×.
2. **It adds essentially nothing on top of them.** `+agg+xseq` (0.695) ≈ `+agg` (0.690), within the
   run-to-run noise. The engineered aggregates already capture the neighbour signal xseq is trying
   to learn.
3. **The mechanism still *works*, it just isn't needed here.** `+xseq` (0.59–0.63) consistently
   beats `local` (0.55–0.58) and the naïve `+meanpool` — so the encoder *does* extract relational
   signal from raw neighbours (as on synthetic); it simply doesn't exceed a rich hand-built summary.

## Why it doesn't replicate — and what it means

The synthetic §8 win was against a **rank-1** memory (a single windowed-velocity scalar): a lossy
summary of a **transient** burst, where seeing the raw neighbours matters. Elliptic's summary is the
opposite: **72 multi-statistic features over the full neighbourhood** (min/max/std/correlations,
including neighbours' own aggregates ≈ 2-hop), summarising a **more static/structural** money-laundering
signal. When the precomputed summary is already rich and the signal is stationary, raw-neighbour
attention has little to add — exactly the boundary the synthetic result implied. This **refines**
the claim rather than overturning it:

> The cross-sequence encoder's advantage scales with how **lossy** the available neighbour summary
> is and how **transient** the relational signal is. It is a large win over a rank-1 memory on
> transient bursts (synthetic); it is redundant with a rich engineered summary on static graph
> structure (Elliptic).

Consistent with our IEEE-CIS finding, GBDTs also dominate the absolute numbers on Elliptic
(LightGBM 0.79–0.82 vs the MLP family 0.55–0.70): tabular, well-featurised fraud favours trees, and
the FFM/attention machinery earns its keep only where the signal is sequential/transient and the
hand-engineered features are lossy.

## Caveats (this is a first pass, not a definitive verdict)

- **Single seed, large variance.** The torch arms swing ±0.03–0.15 run-to-run (no averaging, 25
  epochs, no early stopping); the qualitative ordering (`+agg` ≥ `+xseq`, `+agg+xseq` ≈ `+agg`) is
  stable across both runs, but the exact gaps are not. Multi-seed averaging is needed to firm them.
- **Weak base.** The shared MLP (local 0.58) is far below LightGBM (0.79); the attention comparison
  is fair (same base) but conducted in a weak-encoder regime.
- **Neighbour information is not perfectly matched:** xseq sees ≤16 sampled neighbours' *local*
  features; the aggregated features summarise the *full* neighbourhood (and 2-hop). A matched-receptive-field
  test (give xseq all neighbours, or their aggregated features too) is the honest next step before a
  hard claim that raw attention *cannot* beat the summary here.

Artifacts: `artifacts/elliptic_results.json`. Data: Kaggle `ellipticco/elliptic-data-set`.
