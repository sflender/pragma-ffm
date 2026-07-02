"""Correctness gate: shape checks + overfit a single batch (MLM loss must collapse).

Run:  python -m scripts.smoke
"""
from __future__ import annotations

import torch

from pragma.config import get_preset
from pragma.data.dataset import WindowDataset
from pragma.model.tokenizer import Tokenizer
from pragma.train.pretrain import build_model, mlm_step, to_device
from pragma.utils import count_params, get_device, seed_everything


def main():
    seed_everything(0)
    device = get_device("auto")
    preset = get_preset("nano")
    tok = Tokenizer.load("artifacts/tokenizer.json")

    ds = WindowDataset("data/processed", "train", preset.model.max_seq_len)
    print(f"train windows: {len(ds):,}")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    batch = to_device(next(iter(loader)), device)
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    model = build_model(tok, preset, device)
    print(f"nano params: {count_params(model):,}")

    # shape check
    r = model.record_embeddings(batch["codes"], batch["times"], batch["mask"])
    print(f"record embeddings: {tuple(r.shape)} (expect B,L,d={preset.model.d_model})")
    logits = model.mlm_logits(batch["codes"], batch["times"], batch["mask"])
    print(f"mlm heads: {len(logits)} fields; head0 {tuple(logits[0].shape)}")

    # overfit the single batch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for i in range(120):
        loss, ntok, nreal = mlm_step(model, batch, preset.train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 20 == 0 or i == 119:
            print(f"  step {i:3d} loss {loss.item():.4f}  (masked toks/step ~{int(ntok)})")
        losses.append(loss.item())
    ok = losses[-1] < 0.5 * losses[0]
    print(f"OVERFIT {'PASS' if ok else 'FAIL'}: {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
