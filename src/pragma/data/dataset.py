"""Windowed event-sequence dataset built on the pre-encoded integer arrays.

Rows are pre-sorted by (seq_id, ts). Because ``split`` is a monotone function of
``ts`` and each sequence is time-ordered, the events of one split within a sequence
form a contiguous range -- so we slice per (sequence, split) and tile into fixed-length
windows. Each window is one training example for MLM and one bag of labelled events for
the downstream fraud probe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


class WindowDataset(Dataset):
    def __init__(self, data_dir: str, split: str, max_seq_len: int, min_len: int = 4,
                 stride: int | None = None):
        d = Path(data_dir)
        enc = np.load(d / "encoded.npz")
        self.codes = enc["codes"]          # (N, F) int16, LOCAL ids
        self.ts = enc["ts"]                # (N,) int64 seconds
        self.amount = enc["amount_raw"]    # (N,) float32 raw amount
        self.split_arr = enc["split"]      # (N,) int8
        self.is_fraud = enc["is_fraud"]    # (N,) int8
        self.F = self.codes.shape[1]
        self.L = max_seq_len
        stride = stride or max_seq_len     # non-overlapping by default

        seq = np.load(d / "seq_index.npz")
        code = SPLIT_CODE[split]
        wins = []
        for s, e in zip(seq["starts"], seq["ends"]):
            sub = self.split_arr[s:e]
            idx = np.nonzero(sub == code)[0]
            if idx.size == 0:
                continue
            a, b = s + int(idx[0]), s + int(idx[-1]) + 1   # contiguous range
            for w in range(a, b, stride):
                length = min(self.L, b - w)
                if length >= min_len:
                    wins.append((w, length))
        self.windows = np.asarray(wins, dtype=np.int64)   # (n_windows, 2)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        w, length = int(self.windows[i, 0]), int(self.windows[i, 1])
        L, F = self.L, self.F

        codes = np.zeros((L, F), dtype=np.int64)
        codes[:length] = self.codes[w:w + length]

        ts = self.ts[w:w + length].astype(np.float64)
        times = np.zeros(L, dtype=np.float32)
        times[:length] = (ts - ts[0]) / 86400.0            # days since window start

        mask = np.zeros(L, dtype=bool)
        mask[:length] = True

        fraud = np.full(L, -1, dtype=np.int64)             # -1 = pad / ignore
        fraud[:length] = self.is_fraud[w:w + length]

        amount = np.zeros(L, dtype=np.float32)
        amount[:length] = self.amount[w:w + length]

        return {
            "codes": torch.from_numpy(codes),      # (L, F) long
            "times": torch.from_numpy(times),      # (L,) float
            "mask": torch.from_numpy(mask),        # (L,) bool  True=real
            "fraud": torch.from_numpy(fraud),      # (L,) long  {-1,0,1}
            "amount": torch.from_numpy(amount),    # (L,) float raw amount
        }


class AsOfDateDataset(Dataset):
    """As-of-date (past-only) scoring windows -- the paper's leakage-safe framing.

    For each target transaction, the window is the up-to-L events *ending at* that
    transaction (right-aligned; padding at the front). The bidirectional encoder then
    only ever attends over past events, so no future information enters. The target's
    embedding is read at the final position (index L-1).
    """

    def __init__(self, data_dir: str, targets: np.ndarray, max_seq_len: int):
        d = Path(data_dir)
        enc = np.load(d / "encoded.npz")
        self.codes = enc["codes"]
        self.ts = enc["ts"]
        self.amount = enc["amount_raw"]
        self.is_fraud = enc["is_fraud"]
        self.seq_id = enc["seq_id"]
        self.F = self.codes.shape[1]
        self.L = max_seq_len
        seq = np.load(d / "seq_index.npz")
        self.starts = seq["starts"]                  # per-seq start row
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> dict:
        g = int(self.targets[i])
        s = int(self.starts[self.seq_id[g]])         # sequence start (do not cross)
        a = max(s, g - self.L + 1)
        length = g - a + 1
        L, F = self.L, self.F
        off = L - length                              # right-align

        codes = np.zeros((L, F), dtype=np.int64)
        codes[off:] = self.codes[a:g + 1]

        ts = self.ts[a:g + 1].astype(np.float64)
        times = np.zeros(L, dtype=np.float32)
        times[off:] = (ts - ts[0]) / 86400.0

        mask = np.zeros(L, dtype=bool)
        mask[off:] = True

        amount = np.zeros(L, dtype=np.float32)
        amount[off:] = self.amount[a:g + 1]

        return {
            "codes": torch.from_numpy(codes),
            "times": torch.from_numpy(times),
            "mask": torch.from_numpy(mask),
            "amount": torch.from_numpy(amount),
            "label": torch.tensor(int(self.is_fraud[g]), dtype=torch.long),
        }
