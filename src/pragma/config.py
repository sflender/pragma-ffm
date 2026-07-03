"""Central configuration: model architecture + training presets.

Two model sizes are defined:
  - ``nano``  (~1M params)  : fast iteration, smoke tests, timing benchmarks.
  - ``small`` (~10M params) : headline model, analogue of PRAGMA-S (10M).

Configs are plain dataclasses so they are trivially serialisable and overridable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class TokenizerConfig:
    """Data-encoding hyperparameters. Shared across all model presets (they all read the
    same pre-encoded arrays), so these live here rather than in ModelConfig. This is the
    single source of truth for encode.py; the fitted values are persisted in tokenizer.json.
    """
    n_amount_buckets: int = 64          # percentile buckets for the numerical Amount field
    hash_buckets: int = 4096            # hash space for high-cardinality categoricals
    include_dt: bool = False            # add a log-bucketed time-since-last-event field
    n_dt_buckets: int = 20              # #log-spaced buckets for dt
    dt_min_s: float = 1.0               # smallest dt boundary (seconds)
    dt_max_s: float = 31_536_000.0      # largest dt boundary (~1 year, seconds)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    # --- sequence window ---
    max_seq_len: int = 128              # max events (transactions) per history window

    # --- dimensions ---
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_event_layers: int = 2             # Event Encoder depth (over fields within an event)
    n_history_layers: int = 4           # History Encoder depth (over events in a sequence)
    dropout: float = 0.1

    # --- numeric feature encoding ---
    numeric_mode: str = "bucket"        # bucket | ple | periodic
    periodic_n_freq: int = 16           # #frequencies for the periodic embedding

    # --- rope ---
    rope_theta: float = 10000.0
    # time is measured in *days* since sequence start; scaled before RoPE.
    time_scale_days: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 20000
    grad_clip: float = 1.0
    mask_token_prob: float = 0.15       # token-level masking
    mask_event_prob: float = 0.10       # event-level masking (mask a whole event)
    mask_field_prob: float = 0.10       # key/field-level masking (mask a whole field across events)
    log_every: int = 50
    ckpt_every: int = 2000
    seed: int = 0
    # precision: "fp32" is the safe default on MPS; "bf16" attempted if requested.
    dtype: str = "fp32"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Preset:
    name: str
    model: ModelConfig
    train: TrainConfig


def nano() -> Preset:
    return Preset(
        name="nano",
        model=ModelConfig(
            d_model=96, n_heads=3, d_ff=256,
            n_event_layers=2, n_history_layers=3,
            max_seq_len=96,
        ),
        train=TrainConfig(batch_size=128, max_steps=10000),
    )


def small() -> Preset:
    return Preset(
        name="small",
        model=ModelConfig(
            d_model=256, n_heads=8, d_ff=1024,
            n_event_layers=3, n_history_layers=6,
            max_seq_len=128,
        ),
        train=TrainConfig(batch_size=128, max_steps=30000),
    )


def mini() -> Preset:
    """Mid-point between nano and small for scaling studies (~6-7M params)."""
    return Preset(
        name="mini",
        model=ModelConfig(
            d_model=176, n_heads=4, d_ff=704,
            n_event_layers=3, n_history_layers=5,
            max_seq_len=128,
        ),
        train=TrainConfig(batch_size=128, max_steps=20000),
    )


PRESETS = {"nano": nano, "mini": mini, "small": small}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return PRESETS[name]()
