#!/bin/bash
# IEEE I4 — the load-bearing relational test. Sequence on the client (card1); give the model a
# cross-entity MEMORY over addr1 (region shared across cards). Mirror TabFormer E13 on REAL data:
#   ARM A (no-mem):  embedding-only + duct-tape fusion (FFM emb (+) addr1 features) via fusion_probe
#   ARM B (--mem):   memory-CSA (addr1 cross-attention baked into backbone) via asof_probe
# Question: does learned cross-entity memory close the LGBM gap where a per-sequence model can't?
set -x
export NT="https://ntfy.sh/${NTFY_TOPIC}"
cat > /root/nt.py <<'PYEOF'
import sys, os, shutil, urllib.request
UA={"User-Agent":"Mozilla/5.0"}
def _open(url,data=None,method=None,headers=None,timeout=60):
    h=dict(UA); h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url,data=data,method=method,headers=h),timeout=timeout)
c=sys.argv[1]
try:
    if c=="pub": print(_open(os.environ["NT"],data=sys.argv[2].encode(),timeout=25).read().decode("utf-8","ignore"))
    elif c=="pubfile": _open(os.environ["NT"],data=open(sys.argv[3],"rb").read(),method="PUT",headers={"Filename":sys.argv[2]},timeout=120).read()
    elif c=="dl":
        import time as _t; last=""
        for a in range(5):
            try:
                with open(sys.argv[3],"wb") as fh: shutil.copyfileobj(_open(sys.argv[2],timeout=300),fh)
                print("OK %d"%os.path.getsize(sys.argv[3])); break
            except Exception as e: last="%s:%s"%(type(e).__name__,e); _t.sleep(3*(a+1))
        else: print("DLFAIL %s"%last)
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

STEPS=${MAX_STEPS:-8000}; DIR=data/processed_ieee; TOK=artifacts/tokenizer_ieee.json; PARQ=$DIR/transactions.parquet
pub "POD_UP gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) steps=$STEPS ieee-I4-mem(addr1)"
cd /root; dl "$CODE_URL" code.tgz >/dev/null; mkdir -p pragma && tar --no-same-owner -xzf code.tgz -C pragma
[ -f pragma/pyproject.toml ] || { pub "CODE_FAIL"; term; exit 1; }
cd pragma; pip install -q -e . 2>&1|tail -1; pip install -q lightgbm 2>&1|tail -1
python -c "import torch;print('torch',torch.cuda.get_device_name(0))" 2>&1|while read l; do pub "$l"; done

mkdir -p data/raw/ieee_cis
cat > /root/kag.py <<'PYEOF'
import sys,urllib.request,zipfile,io,os
tok=os.environ["KAGGLE_TOKEN"];f=sys.argv[1]
r=urllib.request.Request("https://www.kaggle.com/api/v1/competitions/data/download/ieee-fraud-detection/"+f,headers={"User-Agent":"Mozilla/5.0","Authorization":"Bearer "+tok})
raw=urllib.request.urlopen(r,timeout=600).read()
try: zipfile.ZipFile(io.BytesIO(raw)).extractall("data/raw/ieee_cis")
except zipfile.BadZipFile: open("data/raw/ieee_cis/"+f,"wb").write(raw)
print("OK",f,len(raw))
PYEOF
pub "fetch kaggle"
python /root/kag.py train_transaction.csv 2>&1|tail -1|while read l; do pub "kag: $l"; done
python /root/kag.py train_identity.csv 2>&1|tail -1|while read l; do pub "kag: $l"; done
[ -f data/raw/ieee_cis/train_transaction.csv ] || { pub "KAGGLE_FAIL"; term; exit 1; }
python -m pragma.data.ieee_cis --raw-dir data/raw/ieee_cis --out $PARQ --seq-key card1 > /root/p.log 2>&1 || { pub "PARSE_FAIL $(tail -3 /root/p.log|tr '\n' ' '|head -c 300)"; term; exit 1; }
python -m pragma.data.encode --dataset ieee_cis --include-dt --parquet $PARQ --out-dir $DIR --tokenizer $TOK > /root/e.log 2>&1 || { pub "ENC_FAIL $(tail -3 /root/e.log|tr '\n' ' '|head -c 300)"; term; exit 1; }
# cross-entity memory over addr1 (region shared across card1 sequences)
python scripts/build_merchant_memory.py --parquet $PARQ --data-dir $DIR --entity addr1 --card-col card1 > /root/mem.log 2>&1 || { pub "MEM_FAIL $(tail -3 /root/mem.log|tr '\n' ' '|head -c 300)"; term; exit 1; }
pub "DATA_READY $(grep -oE 'F=[0-9]+ fields, V=[0-9]+' /root/e.log|head -1) mem=$(grep -oE 'shape=\([0-9, ]+\)' /root/mem.log|head -1)"

train_arm(){  # $1 label  $2 extra(--mem)
  local LB=$1 EX=$2
  pub "TRAIN_${LB} ${EX:-no-mem}"
  local T0=$(date +%s)
  python -u -m pragma.train.pretrain --preset small --numeric-mode bucket --data-dir $DIR --tokenizer $TOK \
    --max-steps $STEPS --batch-size 256 --lr 4.24e-4 --dtype bf16 --num-workers 4 --max-seq-len 128 $EX \
    --tag _$LB --device cuda > /root/tr_$LB.log 2>&1 &
  local PID=$!
  while kill -0 $PID 2>/dev/null; do sleep 300; pub "HB_$LB $(grep -oE 'step +[0-9]+/[0-9]+ loss [0-9.]+' /root/tr_$LB.log|tail -1) | $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|head -1)"; done
  wait $PID
  [ -f artifacts/pretrain_small_bucket_$LB.pt ] || { pub_retry "TRAIN_${LB}_FATAL $(tail -8 /root/tr_$LB.log|tr '\n' ' '|head -c 400)"; return 1; }
  pub "TRAINED_$LB wall=$(( $(date +%s)-T0 ))s"
}

# ARM A: no-mem -> embedding-only + relational-only + duct-tape fusion (entity=addr1)
train_arm A ""
python scripts/fusion_probe.py --ckpt artifacts/pretrain_small_bucket_A.pt --data-dir $DIR --tokenizer $TOK \
  --parquet $PARQ --entity addr1 --card-col card1 --device cuda --batch-size 128 --out artifacts/fusion_A.json > /root/fz.log 2>&1
[ -f artifacts/fusion_A.json ] && { pubfile "fusion_A.json" artifacts/fusion_A.json; pub_retry "RESULT_fusion $(tr -d '\n' < artifacts/fusion_A.json|head -c 800)"; } || pub "FUSION_FAIL $(tail -8 /root/fz.log|tr '\n' ' '|head -c 500)"

# ARM B: --mem -> memory-CSA (addr1 cross-attention)
train_arm B "--mem"
python -m pragma.train.asof_probe --ckpt artifacts/pretrain_small_bucket_B.pt --data-dir $DIR --tokenizer $TOK \
  --device cuda --batch-size 128 --out artifacts/memcsa_B.json > /root/pb.log 2>&1
[ -f artifacts/memcsa_B.json ] && { pubfile "memcsa_B.json" artifacts/memcsa_B.json; pub_retry "RESULT_memcsa $(tr -d '\n' < artifacts/memcsa_B.json|head -c 800)"; } || pub "MEMCSA_FAIL $(tail -8 /root/pb.log|tr '\n' ' '|head -c 500)"
pub "E2E_DONE"
term
