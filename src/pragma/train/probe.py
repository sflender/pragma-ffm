"""Downstream fraud probe: freeze the pretrained backbone, fit a linear classifier on
per-event record embeddings.

This is the PRAGMA "linear probe" adaptation -- the purest test of whether MLM
pretraining produced representations that separate fraud, with no fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataclasses import fields as dc_fields

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pragma.config import ModelConfig
from pragma.data.dataset import WindowDataset
from pragma.eval.metrics import evaluate, print_report
from pragma.model.pragma import MiniPragma
from pragma.model.tokenizer import Tokenizer
from pragma.train.pretrain import to_device
from pragma.utils import get_device, seed_everything


def load_backbone(ckpt_path: str, tok: Tokenizer, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # rebuild ModelConfig from the saved dict, tolerating fields added/removed since.
    valid = {f.name for f in dc_fields(ModelConfig)}
    mcfg = ModelConfig(**{k: v for k, v in ckpt["model_cfg"].items() if k in valid})
    model = MiniPragma(tok, mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("preset", "model"), mcfg


@torch.no_grad()
def extract(model, ds, device, batch_size, max_batches=None, shuffle=False, causal=False):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    embs, labs = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch, device)
        r = model.record_embeddings(batch["codes"], batch["times"], batch["mask"],
                                    batch.get("amount"), causal=causal)
        m = batch["mask"]
        embs.append(r[m].float().cpu().numpy())
        labs.append(batch["fraud"][m].cpu().numpy())
    return np.concatenate(embs), np.concatenate(labs)


def run(ckpt_path, data_dir, tok_path, out_json, train_batches, batch_size,
        device_str="auto", causal=False):
    t0 = time.time()
    device = get_device(device_str)
    seed_everything(0)   # deterministic train-window subsample (probe variance ~±0.02-0.04)
    tok = Tokenizer.load(tok_path)
    model, preset_name, mcfg = load_backbone(ckpt_path, tok, device)
    L = mcfg.max_seq_len
    arm_id = f"{preset_name}_{mcfg.numeric_mode}"

    tr_ds = WindowDataset(data_dir, "train", L)
    te_ds = WindowDataset(data_dir, "test", L)

    Xtr, ytr = extract(model, tr_ds, device, batch_size, max_batches=train_batches,
                       shuffle=True, causal=causal)
    Xte, yte = extract(model, te_ds, device, batch_size, max_batches=None,
                       shuffle=False, causal=causal)
    mode = "causal/as-of-date" if causal else "bidirectional"
    print(f"[probe] {arm_id} mode={mode} train events={len(ytr):,} (fraud={int(ytr.sum()):,}) "
          f"test events={len(yte):,} (fraud={int(yte.sum()):,})  extract {time.time()-t0:.1f}s")

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    scores = clf.predict_proba(scaler.transform(Xte))[:, 1]

    m = evaluate(yte, scores)
    tag = "asof" if causal else "probe"
    stem = Path(ckpt_path).stem.replace("pretrain_", "")   # e.g. nano_bucket_dt
    print_report(f"PRAGMA {stem} {'as-of-date' if causal else 'linear'} probe / test", m)

    result = {"arm": f"pragma_{stem}_{tag}", "ckpt": ckpt_path,
              "causal": causal, "numeric_mode": mcfg.numeric_mode,
              "train_events": int(len(ytr)), "metrics": m}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2))
    print(f"[probe] wrote {out_json}  (total {time.time()-t0:.1f}s)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/pretrain_nano_bucket.pt")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--out", default=None,
                    help="output json; default derived from ckpt name + probe mode")
    ap.add_argument("--train-batches", type=int, default=400,
                    help="number of train windows batches to embed for fitting the probe")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--causal", action="store_true",
                    help="as-of-date: each event's embedding uses only past+self (no future)")
    args = ap.parse_args()
    if args.out is None:
        stem = Path(args.ckpt).stem.replace("pretrain_", "")   # e.g. nano_ple
        suffix = "asof" if args.causal else "probe"
        args.out = f"artifacts/eval_{stem}_{suffix}.json"
    run(args.ckpt, args.data_dir, args.tokenizer, args.out,
        args.train_batches, args.batch_size, args.device, args.causal)


if __name__ == "__main__":
    main()
