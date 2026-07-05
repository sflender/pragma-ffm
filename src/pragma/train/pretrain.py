"""MLM pre-training loop for mini-PRAGMA + reusable step helpers (shared with bench)."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pragma.config import Preset, get_preset
from pragma.data.dataset import WindowDataset
from pragma.model.masking import IGNORE, apply_mlm_mask
from pragma.model.pragma import MiniPragma
from pragma.model.tokenizer import Tokenizer
from pragma.utils import count_params, get_device, seed_everything


def to_device(batch: dict, device) -> dict:
    # synchronous copy: non_blocking races async MPS transfers when a tensor is
    # inspected (.cpu()/.item()) before a model op forces stream ordering.
    return {k: v.to(device) for k, v in batch.items()}


def mlm_loss(logits_list, targets):
    """Mean cross-entropy over masked cells; also returns #predicted tokens."""
    total = torch.zeros((), device=targets.device)
    ntok = torch.zeros((), device=targets.device)
    for j, logits in enumerate(logits_list):
        t = targets[:, :, j].reshape(-1)
        l = logits.reshape(-1, logits.size(-1))
        total = total + F.cross_entropy(l, t, ignore_index=IGNORE, reduction="sum")
        ntok = ntok + (t != IGNORE).sum()
    return total / ntok.clamp(min=1), ntok


def mlm_step(model: MiniPragma, batch: dict, tcfg):
    """One forward pass returning (loss, n_predicted_tokens, n_real_tokens)."""
    codes, times, mask = batch["codes"], batch["times"], batch["mask"]
    masked, targets = apply_mlm_mask(
        codes, mask, tcfg.mask_token_prob, tcfg.mask_event_prob, tcfg.mask_field_prob)
    logits = model.mlm_logits(masked, times, mask, batch.get("amount"))
    loss, ntok = mlm_loss(logits, targets)
    n_real = int(mask.sum().item()) * model.tok.F
    return loss, ntok, n_real


def build_model(tok: Tokenizer, preset: Preset, device) -> MiniPragma:
    model = MiniPragma(tok, preset.model).to(device)
    return model


def cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def train(preset_name: str, data_dir: str, tok_path: str, out_dir: str,
          device_str: str = "auto", max_steps: int | None = None,
          numeric_mode: str | None = None, tag: str = "",
          max_seq_len: int | None = None, pos_mode: str | None = None,
          seed: int | None = None) -> Path:
    preset = get_preset(preset_name)
    tcfg = preset.train
    if max_steps is not None:
        tcfg.max_steps = max_steps
    if numeric_mode is not None:
        preset.model.numeric_mode = numeric_mode
    if max_seq_len is not None:
        preset.model.max_seq_len = max_seq_len
    if pos_mode is not None:
        preset.model.pos_mode = pos_mode
    if seed is not None:
        tcfg.seed = seed
    seed_everything(tcfg.seed)
    device = get_device(device_str)

    tok = Tokenizer.load(tok_path)
    ds = WindowDataset(data_dir, "train", preset.model.max_seq_len)
    loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=0)
    model = build_model(tok, preset, device)
    print(f"[pretrain] preset={preset_name} numeric_mode={preset.model.numeric_mode} "
          f"params={count_params(model):,} device={device} "
          f"windows={len(ds):,} steps={tcfg.max_steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, tcfg.warmup_steps, tcfg.max_steps))

    model.train()
    step, t0, running, skipped = 0, time.time(), 0.0, 0
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"pretrain_{preset_name}_{preset.model.numeric_mode}{tag}.pt"

    while step < tcfg.max_steps:
        for batch in loader:
            batch = to_device(batch, device)
            loss, ntok, _ = mlm_step(model, batch, tcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            # guard: never let a single non-finite gradient poison the weights
            if torch.isfinite(gnorm):
                opt.step()
            else:
                skipped += 1
            sched.step()
            running += loss.item() if torch.isfinite(loss) else 0.0
            step += 1
            if step % tcfg.log_every == 0:
                dt = time.time() - t0
                print(f"  step {step:6d}/{tcfg.max_steps} loss {running/tcfg.log_every:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {tcfg.log_every/dt:.1f} it/s "
                      f"skipped {skipped}")
                running, t0 = 0.0, time.time()
            if step % tcfg.ckpt_every == 0 or step >= tcfg.max_steps:
                torch.save({"model": model.state_dict(), "preset": preset_name,
                            "model_cfg": preset.model.to_dict(), "step": step,
                            "tokenizer": tok_path}, ckpt_path)
            if step >= tcfg.max_steps:
                break
    print(f"[pretrain] saved {ckpt_path}")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--numeric-mode", choices=["bucket", "ple", "periodic"], default=None)
    ap.add_argument("--tag", default="", help="suffix appended to the checkpoint filename")
    ap.add_argument("--max-seq-len", type=int, default=None, help="override context window L")
    ap.add_argument("--pos-mode", choices=["time", "index", "none"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    train(args.preset, args.data_dir, args.tokenizer, args.out_dir, args.device,
          args.max_steps, args.numeric_mode, args.tag, args.max_seq_len,
          args.pos_mode, args.seed)


if __name__ == "__main__":
    main()
