"""Throughput benchmark + full-run time estimate (the local-vs-Bolt decision gate).

Measures sec/step and tokens/sec for each model preset on the active device, then
extrapolates:
  * time per epoch over the full train split,
  * time to reach the preset's configured ``max_steps``.

Run:  python -m pragma.bench.timing --presets nano small
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from pragma.config import get_preset
from pragma.data.dataset import WindowDataset
from pragma.model.tokenizer import Tokenizer
from pragma.train.pretrain import build_model, mlm_step, to_device
from pragma.utils import count_params, get_device


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_preset(name, data_dir, tok, device, warmup, iters):
    preset = get_preset(name)
    tcfg = preset.train
    ds = WindowDataset(data_dir, "train", preset.model.max_seq_len)
    loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, drop_last=True)
    model = build_model(tok, preset, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)

    it = iter(loader)
    batches = []
    for _ in range(warmup + iters):
        try:
            batches.append(to_device(next(it), device))
        except StopIteration:
            it = iter(loader); batches.append(to_device(next(it), device))

    def one(batch):
        loss, _, nreal = mlm_step(model, batch, tcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        return nreal

    for i in range(warmup):
        one(batches[i])
    _sync(device)

    t0 = time.perf_counter()
    tokens = 0
    for i in range(iters):
        tokens += one(batches[warmup + i])
    _sync(device)
    dt = time.perf_counter() - t0

    steps_per_epoch = len(ds) // tcfg.batch_size
    sec_per_step = dt / iters
    return {
        "name": name,
        "params": count_params(model),
        "windows": len(ds),
        "batch": tcfg.batch_size,
        "sec_per_step": sec_per_step,
        "steps_per_sec": 1.0 / sec_per_step,
        "tokens_per_sec": tokens / dt,
        "steps_per_epoch": steps_per_epoch,
        "epoch_min": steps_per_epoch * sec_per_step / 60.0,
        "max_steps": tcfg.max_steps,
        "run_hours": tcfg.max_steps * sec_per_step / 3600.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["nano", "small"])
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    device = get_device(args.device)
    tok = Tokenizer.load(args.tokenizer)
    print(f"device={device}  warmup={args.warmup} iters={args.iters}\n")

    rows = [bench_preset(p, args.data_dir, tok, device, args.warmup, args.iters)
            for p in args.presets]

    hdr = ("preset", "params", "batch", "sec/step", "it/s", "tok/s",
           "steps/epoch", "epoch(min)", "cfg steps", "run(hrs)")
    print("{:<7}{:>11}{:>7}{:>10}{:>7}{:>10}{:>13}{:>12}{:>11}{:>10}".format(*hdr))
    for r in rows:
        print("{:<7}{:>11,}{:>7}{:>10.3f}{:>7.1f}{:>10,.0f}{:>13,}{:>12.1f}{:>11,}{:>10.2f}".format(
            r["name"], r["params"], r["batch"], r["sec_per_step"], r["steps_per_sec"],
            r["tokens_per_sec"], r["steps_per_epoch"], r["epoch_min"], r["max_steps"],
            r["run_hours"]))
    print("\nNote: run(hrs) = configured max_steps x sec/step. "
          "One full epoch over the train split = epoch(min).")


if __name__ == "__main__":
    main()
