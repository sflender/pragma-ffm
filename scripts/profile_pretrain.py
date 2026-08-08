"""Profile the pretraining loop: is it data-bound, compute-bound, or launch-overhead-bound?

Runs three phases on the SAME model/loader the trainer uses, then optionally repeats the compute
phase with ``torch.compile`` so one run answers both "what is the bottleneck?" and "does compiling
help?".

  A data-only      : iterate the DataLoader, touch nothing else      -> input pipeline ceiling
  B compute-only   : one batch held in memory, fwd+bwd repeatedly    -> GPU ceiling
  C end-to-end     : the real loop (load + fwd + bwd + step)         -> what we actually get

Reading it:
  C ~= B  -> compute-bound (compile / bigger batch will help)
  C ~= A  -> input-bound   (more workers / faster storage will help)
  C << both -> per-step overhead (syncs, optimizer, launch latency)

Run: python scripts/profile_pretrain.py --data-dir data/synth_per_card --tokenizer .../tok.json
"""
from __future__ import annotations

import argparse, time
import torch
from torch.utils.data import DataLoader

from pragma.config import get_preset
from pragma.data.dataset import WindowDataset
from pragma.model.tokenizer import Tokenizer
from pragma.train.pretrain import build_model, mlm_step, to_device
from pragma.utils import get_device


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def cycle(loader):
    """Infinite iterator so small datasets don't exhaust mid-measurement."""
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true", help="also time a torch.compile'd model")
    ap.add_argument("--trace", action="store_true", help="print top ops from torch.profiler")
    a = ap.parse_args()

    dev = get_device(a.device)
    tok = Tokenizer.load(a.tokenizer)
    preset = get_preset(a.preset)
    preset.model.numeric_mode = "bucket"
    tcfg = preset.train
    ds = WindowDataset(a.data_dir, "train", preset.model.max_seq_len)
    pin = dev.type == "cuda"
    mk_loader = lambda: DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                                   num_workers=a.num_workers, pin_memory=pin,
                                   persistent_workers=a.num_workers > 0)
    model = build_model(tok, preset, dev)
    use_amp = dev.type == "cuda"
    print(f"[profile] device={dev} preset={a.preset} batch={a.batch_size} workers={a.num_workers} "
          f"windows={len(ds):,} steps={a.steps} (warmup {a.warmup})")

    # ---- A: data-only -------------------------------------------------------
    it = cycle(mk_loader())
    for _ in range(a.warmup):
        next(it)
    t0 = time.time(); n = 0
    for _ in range(a.steps):
        b = next(it); n += 1
    ta = time.time() - t0
    print(f"  A data-only     {n/ta:7.2f} it/s   ({ta/n*1000:6.1f} ms/batch)")

    # ---- B: compute-only ----------------------------------------------------
    def compute_phase(m, label):
        batch = to_device(next(cycle(mk_loader())), dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        for _ in range(a.warmup):                       # includes compile warmup
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _, _ = mlm_step(m, batch, tcfg)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(dev); t = time.time()
        for _ in range(a.steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _, _ = mlm_step(m, batch, tcfg)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(dev); dt = time.time() - t
        print(f"  B compute-only  {a.steps/dt:7.2f} it/s   ({dt/a.steps*1000:6.1f} ms/step)  [{label}]")
        return a.steps / dt

    ips_b = compute_phase(model, "eager")

    # ---- C: end-to-end ------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = cycle(mk_loader()); done = 0
    for _ in range(a.warmup):
        b = to_device(next(it), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, _, _ = mlm_step(model, b, tcfg)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync(dev); t = time.time()
    while done < a.steps:
        b = to_device(next(it), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, _, _ = mlm_step(model, b, tcfg)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        done += 1
    sync(dev); dt = time.time() - t
    ips_c = done / dt
    print(f"  C end-to-end    {ips_c:7.2f} it/s   ({dt/done*1000:6.1f} ms/step)")

    # ---- optional: compiled compute ----------------------------------------
    if a.compile:
        try:
            import torch._dynamo as dynamo
            dynamo.reset(); dynamo.utils.counters.clear()
            cm = build_model(tok, preset, dev)
            # NOTE: MiniPragma defines no forward(); torch.compile(model) would wrap forward()
            # and be bypassed entirely by our custom methods (measured 1.00x = never compiled).
            # Compile the hot SUBMODULES instead -- they are ordinary nn.Modules with forward().
            cm.event = torch.compile(cm.event)
            cm.history = torch.compile(cm.history)
            ips_bc = compute_phase(cm, "compiled-submodules")
            ngraphs = sum(v for k, v in dynamo.utils.counters.get("stats", {}).items()
                          if "graph" in k) or dynamo.utils.counters.get("stats", {})
            print(f"  -> compile speedup on compute: {ips_bc/ips_b:.2f}x  "
                  f"(dynamo stats: {dict(dynamo.utils.counters.get('stats', {}))})")
        except Exception as e:
            print(f"  compile FAILED: {type(e).__name__}: {str(e)[:160]}")

    # ---- verdict ------------------------------------------------------------
    ips_a = n / ta
    print("\n[verdict]")
    print(f"  data ceiling {ips_a:.1f} it/s | compute ceiling {ips_b:.1f} it/s | actual {ips_c:.1f} it/s")
    if ips_c > 0.85 * min(ips_a, ips_b):
        who = "DATA" if ips_a < ips_b else "COMPUTE"
        print(f"  -> {who}-bound (actual is within 15% of the {who.lower()} ceiling)")
    else:
        print("  -> PER-STEP OVERHEAD dominates (actual well below both ceilings): "
              "look at .item() syncs, optimizer, kernel-launch latency")

    if a.trace and dev.type == "cuda":
        from torch.profiler import ProfilerActivity, profile
        batch = to_device(next(cycle(mk_loader())), dev)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _, _ = mlm_step(model, batch, tcfg)
                loss.backward()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
