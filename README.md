# pragma-ffm — a scaled reproduction of a PRAGMA-style financial foundation model

Reproduces, at laptop scale, the core idea of **PRAGMA** (Revolut's financial foundation
model, arXiv:2604.08649): an **encoder-only Transformer pretrained with masked modeling
(MLM) on sequences of banking events**, then adapted to a downstream task via a **linear
probe on frozen embeddings**. We test the paper's central claim — a self-supervised
backbone matches/beats a task-specific incumbent — on an open dataset.

The paper's corpus is proprietary (26M users, 207B tokens). We use the open **IBM
TabFormer** synthetic credit-card dataset (Apache-2.0) and a ~1–14M param model.

## Headline comparison (single task: fraud detection)

| arm | what it is |
|-----|------------|
| **LightGBM** | strong incumbent: gradient-boosted trees on causal engineered features |
| **PRAGMA (linear probe)** | MLM-pretrained backbone, frozen; a linear head on record embeddings |

Metrics: **ROC-AUC** and **PR-AUC (average precision)** — PR-AUC is the headline given
the ~0.12% fraud rate.

## Dataset

IBM TabFormer `card_transaction.v1.csv`: 24.39M transactions, 2000 users, 6139
(user,card) sequences, **0.122% fraud**. Only real label is `Is Fraud?`.

**Split = per-(user,card) sequence** (default, `--split-mode seq`): whole sequences go to
train/val/test — a clean test of generalisation to unseen users, standard in
event-sequence FM work. A `--split-mode temporal` option exists but the TabFormer fraud
rate swings wildly by year (2020 has zero fraud), so temporal is heavily shift-confounded
(LightGBM ROC-AUC drops 0.94 → 0.79). See findings below.

## Model — mini-PRAGMA (`src/pragma/model/`)

Faithful-but-scaled version of the paper's architecture:

1. **Key–value–time tokenizer** (`tokenizer.py`): each event → a set of `(field, value)`
   tokens. Amount is percentile-bucketed (fit on train); low-card categoricals get a
   vocab; high-card (merchant/city/zip) are crc32-hashed; hour/day-of-week derived from
   the timestamp. 10 fields, ~12.8k total value vocab.
2. **Event Encoder** (`encoder.py`): set-transformer over an event's field tokens +
   a learned `[EVT]` token → one event vector.
3. **History Encoder**: bidirectional transformer over the `[EVT]` sequence with **RoPE
   keyed on continuous event times** → a record embedding per event.
4. **MLM heads**: reconstruct masked `(field,value)` tokens from `field_token + record
   embedding`. Masking mixes token/event/field granularities (`masking.py`).

Presets (`config.py`): **nano ~2.9M**, **small ~13.7M** (analogue of PRAGMA-S).

## Pipeline

```bash
# 0. install (torch already present; installs lightgbm/sklearn/pyarrow/tqdm/einops)
pip install -e .

# 1. download data (git-lfs) then extract  (done once)
#    data/raw/TabFormer/data/credit_card/card_transaction.v1.csv

# 2. parse -> typed parquet + splits; then fit tokenizer + pre-encode to int arrays
python -m pragma.data.parse --split-mode seq
python -m pragma.data.encode

# 3. correctness gate (overfit a batch: MLM loss must collapse)
python scripts/smoke.py

# 4. timing / full-run estimate (local-vs-remote decision gate)
python -m pragma.bench.timing --presets nano small

# 5. baseline
python -m pragma.baselines.lgbm

# 6. pretrain (MLM) + downstream probe
python -m pragma.train.pretrain --preset nano
python -m pragma.train.probe --ckpt artifacts/pretrain_nano.pt

# 7. compare all arms
python -m pragma.eval.compare
```

## Environment
M4 Max / 64GB / macOS; miniconda `base` with torch 2.12 (MPS). Throughput (MPS):
nano ≈ 0.48 s/step (~13 min/epoch), small ≈ 2.97 s/step (~60 min/epoch).

## Findings

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for the full lab notebook (hypotheses, per-experiment
setups, caveats, and repro commands). Summary below.

**Headline (per-sequence split, test set = 2.28M transactions, 2,928 frauds):**

| arm | ROC-AUC | PR-AUC | Recall@Prec0.5 |
|-----|---------|--------|----------------|
| LightGBM (causal features) | 0.940 | 0.043 | 0.000 |
| PRAGMA-nano probe, bidirectional | 0.946 | 0.242 | 0.118 |
| **PRAGMA-nano probe, as-of-date (causal, no future)** | **0.944** | **0.215** | **0.092** |

Even a tiny 2.9M-param backbone, pretrained only with MLM and then **frozen** (just a
linear probe on top), **beats the strong LightGBM incumbent by ~5× on PR-AUC** and
achieves nonzero recall at 50% precision where LightGBM gets zero. This reproduces the
paper's central claim at laptop scale.

**Leakage control (the important bit):** PRAGMA is a bidirectional encoder, so a naive
per-window embedding of a transaction sees *future* transactions in the same window —
optimistic for a causal task like fraud. We fixed this to match the paper's "history
truncated at the evaluation point" framing via **as-of-date scoring**: each transaction's
embedding uses only past + itself (causal attention mask). This dropped PR-AUC just
0.242 → 0.215 (~11%), so **the win is real representation quality, not future-peeking**:
0.215 vs LightGBM's 0.043 is still **+396%**, squarely in the spirit of the paper's
reported PR-AUC lifts.

Note on absolute PR-AUC: it looks small because it is anchored to the 0.13% base rate
(random ≈ 0.0013). 0.215 is ~167× better than chance.

**Other notes:**
- TabFormer fraud is **largely a static function** of a few fields (online flag, error
  codes, MCC, amount): LightGBM reaches ROC-AUC 0.94 with essentially one deep tree.
- Temporal split is shift-confounded (LightGBM 0.94 → 0.79); we headline the seq split.

### Numeric-encoding A/B (bucket vs PLE vs periodic)

How the `amount` field is embedded, holding everything else fixed (nano, 3000 steps,
as-of-date probe):

| numeric_mode | ROC-AUC | PR-AUC |
|--------------|---------|--------|
| **bucket** (hard quantile) | 0.944 | **0.219** |
| PLE (piecewise-linear) | 0.942 | 0.110 |
| periodic (Fourier) | 0.941 | 0.087 |

**Surprise: hard quantile bucketing wins decisively on PR-AUC** — the opposite of the
general tabular-DL finding (Gorishniy et al. 2022). ROC-AUC is ~identical; the gap is
entirely in the high-precision region. Leading hypotheses: (1) the MLM target for amount
is the *bucket id*, so bucket **input** is aligned with the reconstruction objective while
PLE/periodic are not; (2) the continuous encoders are freshly-initialised modules,
undertrained at 3000 steps (and saw ~6% guard-skipped steps from early gradient spikes),
vs the shared embedding table bucket reuses; (3) 64 buckets already capture amount's fraud
signal, so extra within-bin resolution doesn't help and the added module dilutes. All
three still beat LightGBM. Caveats: single seed, nano scale, short budget. A fair rematch
would add a continuous (regression) MLM target for the numeric field and more steps.

### Δt (time-since-last-event) embedding

Adding a log-bucketed time-since-previous-event field (`--include-dt`; 20 log-spaced
second buckets), on top of the existing RoPE time encoding and hour/day-of-week features
(bucket mode, nano, 3000 steps, as-of-date probe):

| model | ROC-AUC | PR-AUC | Recall@Prec0.5 |
|-------|---------|--------|----------------|
| bucket (no Δt) | 0.944 | 0.219 | 0.091 |
| **bucket + Δt** | **0.953** | **0.236** | **0.163** |

**Δt helps** — PR-AUC +7.8%, ROC-AUC +0.009, and Recall@Precision-0.5 **+79%** (0.091 →
0.163): at a 50%-precision operating point the model catches ~79% more fraud. The gain
concentrates at high precision, consistent with Δt exposing fraud **bursts** (short gaps)
as directly-usable content — a signal RoPE only encodes implicitly as a pairwise attention
bias. Single-seed caveat applies, but the improvement is consistent across all metrics.

**Training stability:** an MPS softmax-backward NaN (from a `finfo.min` attention mask)
was fixed with a moderate finite mask value (`-1e4`) plus a guard that skips the optimizer
step on any non-finite gradient. nano trains clean (loss 4.87 → 2.04, 0 skips).
