"""Masked-language-model masking for event sequences (PRAGMA-style mixture).

Combines three masking granularities, all restricted to real (non-pad) events:
  * token-level : individual (event, field) cells
  * event-level : all fields of a chosen event
  * field-level : a chosen field across all events in the window

Masked cells have their value replaced by the reserved ``MASK`` id; the target array
holds the original local id at masked cells and ``ignore`` (-100) elsewhere.
"""
from __future__ import annotations

import torch

from pragma.model.tokenizer import MASK

IGNORE = -100


def apply_mlm_mask(codes: torch.Tensor, key_pad: torch.Tensor,
                   p_tok: float, p_evt: float, p_field: float):
    """codes (B,L,F) long, key_pad (B,L) bool True=real -> (masked_codes, targets)."""
    B, L, F = codes.shape
    dev = codes.device
    real = key_pad.unsqueeze(-1)                                   # (B,L,1)

    do = (torch.rand(B, L, F, device=dev) < p_tok)                 # token-level
    do |= (torch.rand(B, L, 1, device=dev) < p_evt)                # event-level
    do |= (torch.rand(B, 1, F, device=dev) < p_field)             # field-level
    do &= real                                                     # never mask padding

    targets = torch.where(do, codes, torch.full_like(codes, IGNORE))
    masked = torch.where(do, torch.full_like(codes, MASK), codes)
    return masked, targets
