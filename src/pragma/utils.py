"""Shared utilities: device selection, seeding, timing."""
from __future__ import annotations

import random
import time

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Pick the best available device. Order: mps -> cuda -> cpu."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map a dtype name to a torch dtype, with MPS-safe fallback."""
    if name == "bf16":
        # bf16 autocast on MPS can be flaky depending on torch build; caller may fall back.
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class Stopwatch:
    """Simple wall-clock timer (context manager)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False
