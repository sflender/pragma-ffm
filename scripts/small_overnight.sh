#!/bin/bash
# Overnight small-model batch: variance + Δt / RoPE / seq-len ablations.
# All small, bucket, 6000 steps, as-of-date probe. Anchor = existing seed-0 baseline (0.494).
cd ~/pragma-ffm
DT="--data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json"      # +Δt (11 fields)
NODT="--data-dir data/processed --tokenizer artifacts/tokenizer.json"          # no Δt (10 fields)
S=6000
TR() { PYTHONUNBUFFERED=1 python -u -m pragma.train.pretrain --preset small --numeric-mode bucket --max-steps $S "$@" 2>&1 | grep -E "numeric_mode|step +6000/|saved"; }
PB() { PYTHONUNBUFFERED=1 python -u -m pragma.train.probe --causal "$@" 2>&1 | grep "as-of-date probe"; }

echo "=== R0: re-probe existing baseline seed0 (seeded probe) ==="
PB --ckpt artifacts/pretrain_small_bucket_dt_6k.pt $DT

echo "=== R1: baseline seed1 (variance) ==="
TR --seed 1 $DT --tag _dt_6k_s1
PB --ckpt artifacts/pretrain_small_bucket_dt_6k_s1.pt $DT

echo "=== R2: ablate Δt (no-dt) seed0 ==="
TR --seed 0 $NODT --tag _6k_nodt
PB --ckpt artifacts/pretrain_small_bucket_6k_nodt.pt $NODT

echo "=== R3: ablate RoPE (pos=none) seed0 ==="
TR --seed 0 --pos-mode none $DT --tag _dt_6k_posnone
PB --ckpt artifacts/pretrain_small_bucket_dt_6k_posnone.pt $DT

echo "=== R4: L=64 seed0 ==="
TR --seed 0 --max-seq-len 64 $DT --tag _dt_6k_L64
PB --ckpt artifacts/pretrain_small_bucket_dt_6k_L64.pt $DT

echo "=== R5: L=256 seed0 ==="
TR --seed 0 --max-seq-len 256 $DT --tag _dt_6k_L256
PB --ckpt artifacts/pretrain_small_bucket_dt_6k_L256.pt $DT

echo "=== SMALL OVERNIGHT DONE ==="
