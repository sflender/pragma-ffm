"""Deterministic held-out MLM loss for a checkpoint.

Low-variance signal for ablations: fixes seed + batch order + masking draws so two
checkpoints are compared on the *identical* masked cells. Reports token-weighted mean
cross-entropy on a held-out split. Reuses the exact training masking + loss path.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from pragma.config import TrainConfig
from pragma.data.dataset import WindowDataset
from pragma.model.tokenizer import Tokenizer
from pragma.train.pretrain import mlm_step, to_device
from pragma.train.probe import load_backbone
from pragma.utils import get_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed_dt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer_dt.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    tok = Tokenizer.load(args.tokenizer)
    model, _, mcfg = load_backbone(args.ckpt, tok, device)
    ds = WindowDataset(args.data_dir, args.split, mcfg.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    tcfg = TrainConfig()                       # default mask probs = training defaults

    seed_everything(0)                         # identical masking across checkpoints
    tot, ntok_tot = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= args.batches:
                break
            b = to_device(b, device)
            loss, ntok, _ = mlm_step(model, b, tcfg)
            n = int(ntok.item())
            tot += float(loss.item()) * n
            ntok_tot += n
    mean = tot / max(1, ntok_tot)
    print(f"[mlm] {args.ckpt}  split={args.split} batches={args.batches} "
          f"loss={mean:.4f} ntok={ntok_tot:,}")


if __name__ == "__main__":
    main()
