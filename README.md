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

### Getting the data

The dataset is **not redistributed here** (2.7 GB raw / ~930 MB processed, and it has a
canonical home). Regenerating it is deterministic, so you get byte-identical arrays:

1. **Download the raw file** from IBM (Apache-2.0): **`github.com/IBM/TabFormer`** →
   `data/credit_card/card_transaction.v1.tgz` (git-lfs; an IBM Box mirror is also linked
   from that repo). Extract to `data/raw/TabFormer/data/credit_card/card_transaction.v1.csv`.
2. **Regenerate the processed arrays** (the Δt variant used for the headline models):
   ```bash
   python -m pragma.data.parse  --split-mode seq          # -> data/processed/ ; seed=0
   python -m pragma.data.encode --include-dt --zip-prefix 0 \
       --out-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json
   ```
   > **Note (E12):** the *default* zip encoding is now the region-preserving **3-digit prefix**
   > (`TokenizerConfig.zip_prefix=3`). The **v0.1 release checkpoints were trained on the older
   > *hashed* zip encoding**, so reproducing them byte-for-byte requires **`--zip-prefix 0`** (as
   > above, which disables the prefix and restores hashing). New training runs (omit the flag) use
   > the improved prefix encoding.

**Why this reproduces exactly:** the train/val/test assignment is a seeded permutation of
the 6,139 `(user,card)` sequences (`np.random.default_rng(seed=0)` in `parse.py`), and the
tokenizer (vocab / hashing / quantile buckets, fit on train only) is committed to the repo
(`artifacts/tokenizer_dt.json`) — so the splits and encodings are identical across machines.
Please cite **Padhi et al. 2020 (TabFormer)** and keep the Apache-2.0 license if you
redistribute any derived copy.

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

# 1. get the data (see "Getting the data": download from github.com/IBM/TabFormer ->
#    data/raw/TabFormer/data/credit_card/card_transaction.v1.csv)

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

## Pretrained models & reproduction

The trained weights are published as **GitHub release assets** (tag `v0.1`) — they're too
large for git (see `.gitignore`). Tokenizers (`artifacts/tokenizer*.json`) and result JSONs
are committed to the repo, so you only need to fetch the weights.

**Release contents (`v0.1`):**

| asset | what it is |
|-------|------------|
| `pretrain_nano_bucket_dt.pt` | nano FFM (2.9M), MLM-pretrained, bucket + Δt |
| `pretrain_mini_bucket_dt.pt` | mini FFM (7.5M) |
| `pretrain_small_bucket_dt_6k.pt` | small FFM (13.7M), 6k steps — the headline model |
| `lgbm_model.txt` | LightGBM baseline booster (native format) |

Each `.pt` embeds its own `model_cfg`/`preset`/`step`. The matching tokenizer is
`artifacts/tokenizer_dt.json` (already in the repo).

**A. Reproduce evals from the published weights (fast — no pretraining):**

```bash
pip install -e .
python scripts/fetch_weights.py                 # pulls the 3 FFMs + lgbm_model.txt into artifacts/

# you still need the (large, regenerable) dataset — see step B1–B2 below, then:
python -m pragma.train.asof_probe --ckpt artifacts/pretrain_small_bucket_dt_6k.pt \
    --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json
python scripts/lgbm_subsample.py                 # LightGBM baseline on the same subsample
python scripts/make_figures.py                   # regenerate figures/fig1..7
```

Load a checkpoint in code:

```python
from pragma.model.tokenizer import Tokenizer
from pragma.train.probe import load_backbone
from pragma.utils import get_device
tok = Tokenizer.load("artifacts/tokenizer_dt.json")
model, preset, cfg = load_backbone("artifacts/pretrain_small_bucket_dt_6k.pt", tok, get_device("auto"))
# model.record_embeddings(codes, times, mask, amount, causal=False) -> per-event embeddings
```

**B. Reproduce end-to-end from scratch** (the data is *not* redistributed — regenerate it):

```bash
# B1. download IBM TabFormer (Apache-2.0) from github.com/IBM/TabFormer ->
#     data/raw/TabFormer/data/credit_card/card_transaction.v1.csv   (see "Getting the data")
# B2. parse -> parquet + deterministic per-(user,card) split, then fit tokenizer + encode
python -m pragma.data.parse  --split-mode seq
python -m pragma.data.encode --include-dt --zip-prefix 0 --out-dir data/processed_dt \
    --tokenizer artifacts/tokenizer_dt.json          # -> data/processed_dt/ (--zip-prefix 0 = v0.1 hashed zip)
# B3. pretrain (small = 6k steps ~2h on M4 Max; nano/mini = 3k)
python -m pragma.train.pretrain --preset small --numeric-mode bucket --max-steps 6000 \
    --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json --tag _dt_6k
# B4. eval + baseline + figures  (as in A)
```

Reproducing the exact published weights bit-for-bit isn't guaranteed (MPS nondeterminism),
but the metrics reproduce within seed noise. See **EXPERIMENTS.md** for every experiment's
exact command.

## Environment
M4 Max / 64GB / macOS; miniconda `base` with torch 2.12 (MPS). Throughput (MPS):
nano ≈ 0.48 s/step (~13 min/epoch), small ≈ 2.97 s/step (~60 min/epoch).

## Findings

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for the full lab notebook (hypotheses, per-experiment
setups, caveats, and repro commands). Summary below.

> **Note on eval methodology (E10).** The headline numbers below use the **corrected
> as-of-date evaluation** (method a): for each target transaction we run the backbone
> bidirectionally over a window *ending at* that transaction and read the last-position
> embedding — the PRAGMA-faithful scheme. Earlier revisions of this README reported an
> older tiling eval on the full test set; those absolute numbers are superseded. **PR-AUC
> here is on a stratified eval subsample (~1.9% base rate), so treat PR-AUC as a
> *relative* metric across arms and ROC-AUC (base-rate-invariant) as the absolute one.**

**Headline (per-sequence split, as-of-date eval, ~1.9% base-rate subsample):**

| arm | PR-AUC | vs LightGBM |
|-----|--------|-------------|
| LightGBM (causal engineered features) | 0.369 | — |
| PRAGMA **nano** (2.9M) probe | 0.643 | **+74%** |
| PRAGMA **mini** (7.5M) probe | 0.695 | +88% |
| PRAGMA **small** (13.7M) probe | **0.786** | **+113%** |

A frozen, MLM-pretrained backbone with only a **linear probe** on top beats the strong
LightGBM incumbent by **+74% (nano) to +113% (small)** on PR-AUC, and **scales cleanly**
with model size (0.643 → 0.695 → 0.786). This reproduces the paper's central claim at
laptop scale. Ablations (`figures/fig5_ablations.png`): **RoPE is load-bearing** (−7.2
PR-AUC points); Δt (−0.6) and the field embedding (~0) are near-neutral (redundant given
RoPE / the value-id offsets). See EXPERIMENTS.md E1–E11.

**Caveats (read before citing):** single **synthetic** dataset with rule-injected fraud
(likely more learnable than real fraud; simulated span **1991–2020**); mostly **single-seed**
(MLM-loss seed variance ~0.2 — small deltas are noise); PR-AUC on a subsample as noted.

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
