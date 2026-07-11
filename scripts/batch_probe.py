"""GPU throughput probe: sweep batch sizes for a preset, find what fits and what's fastest,
and measure how much of a step is data-loading vs compute (the "is the data path a
bottleneck?" question).

Run:  python scripts/batch_probe.py --preset small --data-dir data/processed_dt \
          --tokenizer artifacts/tokenizer_dt.json --sizes 128 256 512 1024 2048
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
from pragma.utils import get_device


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def probe_size(preset, tok, ds, device, bs, iters=12, warmup=4):
    """One batch size: peak GPU mem, sec/step, samples/s. Returns None if OOM."""
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = build_model(tok, preset, device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    it = iter(loader)
    try:
        def one():
            b = to_device(next(it), device, non_blocking=device.type == "cuda")
            loss, _, _ = mlm_step(model, b, preset.train)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for _ in range(warmup):
            one()
        _sync(device); t0 = time.perf_counter()
        for _ in range(iters):
            one()
        _sync(device); dt = (time.perf_counter() - t0) / iters
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        return {"bs": bs, "sec_step": dt, "samples_s": bs / dt, "peak_gb": peak}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return None
        raise
    finally:
        del model, opt


def data_path_timing(preset, tok, ds, device, bs, iters=20):
    """Split a step into: fetch+H2D transfer vs GPU compute (fwd+bwd+opt)."""
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = build_model(tok, preset, device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    it = iter(loader)
    for _ in range(4):  # warmup
        b = to_device(next(it), device, non_blocking=device.type == "cuda")
        loss, _, _ = mlm_step(model, b, preset.train); opt.zero_grad(); loss.backward(); opt.step()
    _sync(device)
    t_data = t_compute = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        raw = next(it); b = to_device(raw, device, non_blocking=device.type == "cuda"); _sync(device)
        t1 = time.perf_counter()
        loss, _, _ = mlm_step(model, b, preset.train); opt.zero_grad(); loss.backward(); opt.step(); _sync(device)
        t2 = time.perf_counter()
        t_data += t1 - t0; t_compute += t2 - t1
    tot = t_data + t_compute
    return {"data_ms": 1e3 * t_data / iters, "compute_ms": 1e3 * t_compute / iters,
            "data_frac": t_data / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--data-dir", default="data/processed_dt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer_dt.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    args = ap.parse_args()

    device = get_device(args.device)
    tok = Tokenizer.load(args.tokenizer)
    preset = get_preset(args.preset)
    ds = WindowDataset(args.data_dir, "train", preset.model.max_seq_len)
    print(f"[probe] preset={args.preset} L={preset.model.max_seq_len} device={device} "
          f"windows={len(ds):,}")

    best = None
    for bs in args.sizes:
        r = probe_size(preset, tok, ds, device, bs)
        if r is None:
            print(f"  bs={bs:>5}  OOM"); break
        print(f"  bs={bs:>5}  {r['sec_step']*1e3:7.1f} ms/step  "
              f"{r['samples_s']:8.0f} samples/s  peak {r['peak_gb']:.1f} GB")
        if best is None or r["samples_s"] > best["samples_s"]:
            best = r
    if best:
        print(f"[probe] throughput-optimal: bs={best['bs']} "
              f"({best['samples_s']:.0f} samples/s, {best['peak_gb']:.1f} GB)")
        dp = data_path_timing(preset, tok, ds, device, best["bs"])
        print(f"[probe] data-path @bs={best['bs']}: fetch+H2D {dp['data_ms']:.1f} ms | "
              f"compute {dp['compute_ms']:.1f} ms | data = {dp['data_frac']*100:.1f}% of step")
        print(f"BEST_BS={best['bs']}")   # machine-readable for the training driver


if __name__ == "__main__":
    main()
