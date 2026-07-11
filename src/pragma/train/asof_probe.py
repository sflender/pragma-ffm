"""Sliding-window as-of-date probe (method a) -- the PRAGMA-faithful evaluation.

For each target transaction we build a window ending at it (up to L events, the target
last), run the frozen backbone BIDIRECTIONALLY (no causal mask -- there is no future in
the window), and read the last-position embedding. The same extraction is used for BOTH
the probe's train set and its test set, so the logistic head is fit and scored on the same
kind of embedding.

Targets are stratified (all fraud + a capped non-fraud sample); use the SAME caps across
arms so PR-AUC is comparable. Reuses saved checkpoints -- no retraining.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pragma.data.dataset import AsOfDateDataset
from pragma.eval.metrics import evaluate, print_report
from pragma.model.tokenizer import Tokenizer
from pragma.train.pretrain import to_device
from pragma.train.probe import load_backbone
from pragma.utils import get_device, seed_everything

SPLIT = {"train": 0, "val": 1, "test": 2}


def build_targets(data_dir, split, rng, max_neg):
    """All fraud rows of the split + a capped random non-fraud sample."""
    enc = np.load(Path(data_dir) / "encoded.npz")
    idx = np.nonzero(enc["split"] == SPLIT[split])[0]
    y = enc["is_fraud"][idx]
    pos, neg = idx[y == 1], idx[y == 0]
    if max_neg is not None and neg.size > max_neg:
        neg = rng.choice(neg, size=max_neg, replace=False)
    out = np.concatenate([pos, neg])
    rng.shuffle(out)
    return out


@torch.no_grad()
def embed(model, ds, device, batch_size):
    """Last-position (target) embedding for each as-of-date window."""
    embs, labs = [], []
    for b in DataLoader(ds, batch_size=batch_size):
        b = to_device(b, device)
        r = model.record_embeddings(b["codes"], b["times"], b["mask"], b["amount"], causal=False)
        embs.append(r[:, -1].float().cpu().numpy())
        labs.append(b["label"].cpu().numpy())
    return np.concatenate(embs), np.concatenate(labs)


def run(ckpt, data_dir, tok_path, out_json, train_max_neg, test_max_neg, batch_size, device_str,
        window_len=None):
    t0 = time.time()
    device = get_device(device_str)
    seed_everything(0)
    tok = Tokenizer.load(tok_path)
    model, preset_name, mcfg = load_backbone(ckpt, tok, device)
    L = window_len or mcfg.max_seq_len          # allow overriding the eval window length
    rng = np.random.default_rng(0)

    tr = build_targets(data_dir, "train", rng, train_max_neg)
    te = build_targets(data_dir, "test", rng, test_max_neg)
    Xtr, ytr = embed(model, AsOfDateDataset(data_dir, tr, L), device, batch_size)
    Xte, yte = embed(model, AsOfDateDataset(data_dir, te, L), device, batch_size)
    print(f"[swasof] train {len(ytr):,} (fraud {int(ytr.sum()):,}) | "
          f"test {len(yte):,} (fraud {int(yte.sum()):,}, base {yte.mean():.4f}) | {time.time()-t0:.0f}s")

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    scores = clf.predict_proba(scaler.transform(Xte))[:, 1]

    m = evaluate(yte, scores)
    stem = Path(ckpt).stem.replace("pretrain_", "")
    print_report(f"[a/sliding-window] {stem}", m)
    result = {"arm": f"pragma_{stem}_swasof", "ckpt": ckpt, "eval": "sliding_window_asof",
              "L": L, "test_base_rate": float(yte.mean()), "metrics": m}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2))
    print(f"[swasof] wrote {out_json}  (total {time.time()-t0:.0f}s)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/processed_dt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer_dt.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-max-neg", type=int, default=150000)
    ap.add_argument("--test-max-neg", type=int, default=150000)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="embedding batch size; lower to ~128 if the GPU OOMs (d_model=256 small). "
                         "Does not affect results — embeddings are per-window.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--window-len", type=int, default=None,
                    help="override eval window length (default = model's max_seq_len)")
    args = ap.parse_args()
    if args.out is None:
        stem = Path(args.ckpt).stem.replace("pretrain_", "")
        wl = f"_w{args.window_len}" if args.window_len else ""
        args.out = f"artifacts/eval_{stem}{wl}_swasof.json"
    run(args.ckpt, args.data_dir, args.tokenizer, args.out,
        args.train_max_neg, args.test_max_neg, args.batch_size, args.device, args.window_len)


if __name__ == "__main__":
    main()
