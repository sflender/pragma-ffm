#!/bin/bash
# Field-embedding ablation (E11): small model, field_emb OFF, matched to the existing
# small_bucket_dt_6k baseline (field_emb ON). Sequential — NO concurrent GPU jobs.
#   1. pretrain small, bucket, +dt, 6000 steps, seed 0, --no-field-emb
#   2. as-of-date probe (method a)  -> eval_small_bucket_dt_6k_nofe_swasof.json
#   3. deterministic held-out MLM loss for BOTH checkpoints (identical masking) -> comparison
# Baseline (field_emb ON): PR-AUC 0.786, val MLM loss ~1.36.
set -u
cd ~/pragma-ffm
DT="--data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json"
BASE=artifacts/pretrain_small_bucket_dt_6k.pt
ABL=artifacts/pretrain_small_bucket_dt_6k_nofe.pt
RES=/tmp/fieldemb_results.txt
: > "$RES"

echo "=== [E11] $(date) START: pretrain small --no-field-emb (seed 0, 6000 steps) ==="
PYTHONUNBUFFERED=1 python -u -m pragma.train.pretrain --preset small --numeric-mode bucket \
  --max-steps 6000 $DT --seed 0 --no-field-emb --tag _dt_6k_nofe 2>&1 \
  | grep -E "preset=|step +6000/|step +5[0-9]{3}/|saved|Error|Traceback"

echo "=== [E11] $(date) as-of-date probe (method a) ==="
PYTHONUNBUFFERED=1 python -u -m pragma.train.asof_probe --ckpt "$ABL" $DT 2>&1 \
  | grep -E "swasof|sliding-window|Error|Traceback"

echo "=== [E11] $(date) held-out MLM loss (identical masking, 100 val batches) ==="
BASE_MLM=$(python -u scripts/eval_mlm.py --ckpt "$BASE" --batches 100 2>&1 | grep '\[mlm\]')
ABL_MLM=$(python -u scripts/eval_mlm.py --ckpt "$ABL"  --batches 100 2>&1 | grep '\[mlm\]')

echo "=== [E11] $(date) DONE — COMPARISON ===" | tee -a "$RES"
{
  echo "field_emb ON  (baseline): $(python -c "import json;print('PR-AUC',round(json.load(open('artifacts/eval_small_bucket_dt_6k_swasof.json'))['metrics']['pr_auc'],4))")"
  echo "field_emb OFF (ablation): $(python -c "import json;print('PR-AUC',round(json.load(open('artifacts/eval_small_bucket_dt_6k_nofe_swasof.json'))['metrics']['pr_auc'],4))" 2>&1)"
  echo "MLM ON : $BASE_MLM"
  echo "MLM OFF: $ABL_MLM"
} | tee -a "$RES"
echo "=== [E11] results also in $RES ==="
