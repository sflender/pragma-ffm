#!/bin/bash
# S1 — the controlled relational test. Synthetic data with KNOWN fraud structure:
#   per_card   : fraud is a burst on the card's own history (per-sequence FFM should detect)
#   relational : compromised-merchant velocity spike (per-sequence FFM should FAIL; memory recovers)
# Per mode: no-mem FFM -> fusion_probe (embedding-only + duct-tape merchant feats);
#           +mem FFM (d_mem=7, windowed velocity) -> memory-CSA probe. Self-contained (no Kaggle).
set -x
export NT="https://ntfy.sh/${NTFY_TOPIC}"
cat > /root/nt.py <<'PYEOF'
import sys, os, urllib.request
UA={"User-Agent":"Mozilla/5.0"}
def _open(url,data=None,method=None,headers=None,timeout=60):
    h=dict(UA); h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url,data=data,method=method,headers=h),timeout=timeout)
c=sys.argv[1]
try:
    if c=="pub": print(_open(os.environ["NT"],data=sys.argv[2].encode(),timeout=25).read().decode("utf-8","ignore"))
    elif c=="pubfile": _open(os.environ["NT"],data=open(sys.argv[3],"rb").read(),method="PUT",headers={"Filename":sys.argv[2]},timeout=120).read()
    elif c=="dl":
        import shutil,time as _t
        for a in range(5):
            try:
                with open(sys.argv[3],"wb") as fh: shutil.copyfileobj(_open(sys.argv[2],timeout=300),fh)
                print("OK %d"%os.path.getsize(sys.argv[3])); break
            except Exception as e: _t.sleep(3*(a+1))
    elif c=="term":
        b=('{"query":"mutation{podTerminate(input:{podId:\\"%s\\"})}"}'%os.environ.get("RUNPOD_POD_ID","")).encode()
        _open("https://api.runpod.io/graphql?api_key=%s"%os.environ["RUNPOD_API_KEY"],data=b,headers={"Content-Type":"application/json"},timeout=30).read()
except Exception as e: sys.stderr.write("nt %s FAIL %s\n"%(c,e))
PYEOF
pub(){ python /root/nt.py pub "$1" 2>/dev/null || true; }
pubfile(){ python /root/nt.py pubfile "$1" "$2" 2>/dev/null || true; }
dl(){ python /root/nt.py dl "$1" "$2"; }
pub_retry(){ for w in 0 5 15 45 90; do sleep $w; if python /root/nt.py pub "$1" 2>/dev/null|grep -q '"id"'; then return 0; fi; done; }
term(){ python /root/nt.py term 2>/dev/null || true; }
trap term EXIT
( sleep 14400; pub "WATCHDOG 4h"; term ) &

STEPS=${MAX_STEPS:-6000}
pub "POD_UP gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) steps=$STEPS synth-S1"
cd /root; dl "$CODE_URL" code.tgz >/dev/null; mkdir -p pragma && tar --no-same-owner -xzf code.tgz -C pragma
[ -f pragma/pyproject.toml ] || { pub "CODE_FAIL"; term; exit 1; }
cd pragma; pip install -q -e . 2>&1|tail -1; pip install -q lightgbm 2>&1|tail -1
python -c "import torch;print('torch',torch.cuda.get_device_name(0))" 2>&1|while read l; do pub "$l"; done

run_mode(){  # $1 = mode (per_card|relational)
  local MODE=$1 D=data/synth_$1 TOK=artifacts/tok_$1.json PARQ=data/synth_$1/transactions.parquet
  pub "MODE_${MODE}_START"
  python scripts/gen_synth_relational.py --mode $MODE --out $PARQ > /root/g_$MODE.log 2>&1 || { pub "${MODE}_GEN_FAIL"; return 1; }
  python -m pragma.data.encode --dataset synth --include-dt --parquet $PARQ --out-dir $D --tokenizer $TOK > /root/e_$MODE.log 2>&1 || { pub "${MODE}_ENC_FAIL"; return 1; }
  python scripts/build_merchant_memory.py --parquet $PARQ --data-dir $D --entity merchant --card-col card --windows 3600,900 > /root/m_$MODE.log 2>&1 || { pub "${MODE}_MEM_FAIL"; return 1; }
  pub "${MODE}_DATA fraud=$(grep -oE 'fraud [0-9.]+' /root/g_$MODE.log|head -1) mem=$(grep -oE 'shape=\([0-9, ]+\)' /root/m_$MODE.log|head -1)"

  # ARM A: no-mem -> fusion_probe (embedding-only + relational-only + duct-tape)
  python -u -m pragma.train.pretrain --preset small --numeric-mode bucket --data-dir $D --tokenizer $TOK \
    --max-steps $STEPS --batch-size 256 --lr 4.24e-4 --dtype bf16 --num-workers 4 --max-seq-len 128 \
    --tag _${MODE}_A --device cuda > /root/trA_$MODE.log 2>&1 &
  local PA=$!; while kill -0 $PA 2>/dev/null; do sleep 300; pub "HB_${MODE}_A $(grep -oE 'step +[0-9]+/[0-9]+ loss [0-9.]+' /root/trA_$MODE.log|tail -1)"; done; wait $PA
  local CKA=artifacts/pretrain_small_bucket_${MODE}_A.pt
  [ -f "$CKA" ] || { pub_retry "${MODE}_A_FATAL $(tail -6 /root/trA_$MODE.log|tr '\n' ' '|head -c 400)"; return 1; }
  python scripts/fusion_probe.py --ckpt "$CKA" --data-dir $D --tokenizer $TOK --parquet $PARQ \
    --entity merchant --card-col card --device cuda --batch-size 128 --out artifacts/fus_$MODE.json > /root/fzA_$MODE.log 2>&1
  [ -f artifacts/fus_$MODE.json ] && { pubfile "fus_$MODE.json" artifacts/fus_$MODE.json; pub_retry "RESULT_fusion_${MODE} $(tr -d '\n' < artifacts/fus_$MODE.json|head -c 700)"; } || pub "${MODE}_FUS_FAIL $(tail -6 /root/fzA_$MODE.log|tr '\n' ' '|head -c 400)"

  # ARM B: +mem (d_mem=7 windowed velocity) -> memory-CSA
  python -u -m pragma.train.pretrain --preset small --numeric-mode bucket --data-dir $D --tokenizer $TOK \
    --max-steps $STEPS --batch-size 256 --lr 4.24e-4 --dtype bf16 --num-workers 4 --max-seq-len 128 --mem --d-mem 7 \
    --tag _${MODE}_B --device cuda > /root/trB_$MODE.log 2>&1 &
  local PB=$!; while kill -0 $PB 2>/dev/null; do sleep 300; pub "HB_${MODE}_B $(grep -oE 'step +[0-9]+/[0-9]+ loss [0-9.]+' /root/trB_$MODE.log|tail -1)"; done; wait $PB
  local CKB=artifacts/pretrain_small_bucket_${MODE}_B.pt
  [ -f "$CKB" ] || { pub_retry "${MODE}_B_FATAL $(tail -6 /root/trB_$MODE.log|tr '\n' ' '|head -c 400)"; return 1; }
  python -m pragma.train.asof_probe --ckpt "$CKB" --data-dir $D --tokenizer $TOK --device cuda --batch-size 128 \
    --out artifacts/memcsa_$MODE.json > /root/pbB_$MODE.log 2>&1
  [ -f artifacts/memcsa_$MODE.json ] && { pubfile "memcsa_$MODE.json" artifacts/memcsa_$MODE.json; pub_retry "RESULT_memcsa_${MODE} $(tr -d '\n' < artifacts/memcsa_$MODE.json|head -c 700)"; } || pub "${MODE}_MEMCSA_FAIL $(tail -6 /root/pbB_$MODE.log|tr '\n' ' '|head -c 400)"
}

run_mode relational
run_mode per_card
pub "E2E_DONE"
term
