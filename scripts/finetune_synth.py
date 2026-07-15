"""End-to-end supervised fine-tuning — the 'unfreeze + aligned objective' test.

S1 showed the frozen MLM backbone + linear probe cannot extract cross-entity signal even when
the exact windowed-velocity feature is in the memory (memory-CSA PR 0.047 vs LGBM 0.36). The
proxy-alignment analysis says the fix is to (a) align the objective (train on the fraud label,
not MLM) and (b) unfreeze (let the head shape the backbone). This trains the SAME architecture
(optionally with memory) end-to-end on the fraud label over as-of-date windows and reports
PR-AUC — directly testing whether the signal becomes extractable.

Run: python scripts/finetune_synth.py --data-dir data/synth_rel --tokenizer artifacts/tok_synth_rel.json --mem --d-mem 7
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from pragma.config import get_preset
from pragma.data.dataset import AsOfDateDataset
from pragma.model.pragma import MiniPragma
from pragma.model.tokenizer import Tokenizer
from pragma.train.asof_probe import build_targets
from pragma.train.pretrain import to_device
from pragma.utils import get_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--d-mem", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-neg", type=int, default=150000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out", default="artifacts/finetune_synth.json")
    args = ap.parse_args()

    t0 = time.time()
    device = get_device(args.device); seed_everything(0)
    tok = Tokenizer.load(args.tokenizer)
    preset = get_preset(args.preset)
    preset.model.numeric_mode = "bucket"; preset.model.use_mem = args.mem
    if args.mem: preset.model.d_mem = args.d_mem
    L = preset.model.max_seq_len
    model = MiniPragma(tok, preset.model).to(device)
    head = nn.Linear(preset.model.d_model, 1).to(device)
    use_amp = args.dtype == "bf16" and device.type == "cuda"

    rng = np.random.default_rng(0)
    tr = build_targets(args.data_dir, "train", rng, args.max_neg)
    te = build_targets(args.data_dir, "test", rng, args.max_neg)
    tr_ds, te_ds = AsOfDateDataset(args.data_dir, tr, L), AsOfDateDataset(args.data_dir, te, L)
    # class weight for the BCE (fraud is rare)
    ytr_all = np.array([tr_ds[i]["label"] for i in range(len(tr_ds))]) if False else None
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=0.01)

    def fwd(b):
        r = model.record_embeddings(b["codes"], b["times"], b["mask"], b["amount"],
                                    causal=False, mem=b.get("mem"))
        return head(r[:, -1].float()).squeeze(-1)

    # estimate pos_weight from a pass over train labels
    pw = None
    for b in DataLoader(tr_ds, batch_size=512):
        y = b["label"].numpy(); pos = int(y.sum()); neg = int((y == 0).sum())
        pw = torch.tensor([max(1.0, neg / max(1, pos))], device=device); break
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)

    model.train()
    for ep in range(args.epochs):
        tot = 0.0; n = 0
        for b in DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True):
            b = to_device(b, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logit = fwd(b); loss = lossfn(logit, b["label"].float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"[ft] epoch {ep+1}/{args.epochs} loss {tot/max(1,n):.4f} ({time.time()-t0:.0f}s)")

    model.eval(); scores = []; labs = []
    with torch.no_grad():
        for b in DataLoader(te_ds, batch_size=256):
            b = to_device(b, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                s = torch.sigmoid(fwd(b))
            scores.append(s.float().cpu().numpy()); labs.append(b["label"].cpu().numpy())
    s = np.concatenate(scores); y = np.concatenate(labs)
    res = {"arm": f"finetune_{'mem' if args.mem else 'nomem'}", "data_dir": args.data_dir,
           "n": int(len(y)), "base": float(y.mean()), "epochs": args.epochs,
           "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}
    print(f"[ft] RESULT mem={args.mem} PR {res['pr_auc']:.3f} ROC {res['roc_auc']:.3f} base {res['base']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
