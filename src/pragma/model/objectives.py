"""Alternative self-supervised pretraining objectives for the FFM.

Baseline is masked modelling (``pragma.model.masking``) with plain per-field cross-entropy.
This module adds two research variants:

* **ordinal MLM** (``ordinal_mlm_loss``) -- numeric fields are percentile *buckets*, i.e. an
  ordered scale, but plain cross-entropy treats them as unordered classes: predicting bucket 7
  when the truth is 8 is penalised exactly as much as predicting bucket 40. We replace the
  one-hot target on numeric-like fields with an exponentially-decaying soft target over
  neighbouring buckets, so near-misses are cheap and the head learns the ordering.

* **ELECTRA-style replaced-token detection** (``apply_rtd_corruption`` / ``rtd_loss``) --
  instead of blanking cells with ``[MASK]`` and reconstructing them, corrupt a fraction of cells
  with *plausible* values sampled from each field's empirical marginal, and ask a binary head at
  **every** cell "was this replaced?". Two consequences: every position produces gradient (vs.
  ~31% under MLM), and no ``[MASK]`` token ever appears, removing the pretrain/fine-tune input
  mismatch. The objective ("is this event plausible?") is also much closer to fraud detection
  than reconstruction is.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from pragma.model.masking import IGNORE
from pragma.model.tokenizer import N_SPECIAL

ORDINAL_KINDS = ("num", "dtlog")


# ------------------------------------------------------------------ ordinal MLM
def ordinal_mlm_loss(logits_list, targets, fields, tau: float = 1.0):
    """Per-field CE, but ordinal-aware soft targets on numeric-like fields.

    ``fields`` are the tokenizer FieldSpecs (same order as ``logits_list``). ``tau`` is the
    decay width in buckets: weight(i) ~ exp(-|i - b| / tau) over the non-special bucket range.
    tau -> 0 recovers plain cross-entropy.
    """
    total = torch.zeros((), device=targets.device)
    ntok = torch.zeros((), device=targets.device)
    for j, logits in enumerate(logits_list):
        t = targets[:, :, j].reshape(-1)                      # (N,)
        l = logits.reshape(-1, logits.size(-1))               # (N,V_j)
        keep = t != IGNORE
        if not bool(keep.any()):
            continue
        if fields[j].kind not in ORDINAL_KINDS:
            total = total + F.cross_entropy(l, t, ignore_index=IGNORE, reduction="sum")
        else:
            lk, tk = l[keep], t[keep]                         # (M,V_j), (M,)
            V = lk.size(-1)
            idx = torch.arange(V, device=lk.device).unsqueeze(0)          # (1,V)
            w = torch.exp(-(idx - tk.unsqueeze(1)).abs().float() / tau)   # (M,V)
            w[:, :N_SPECIAL] = 0.0                            # never put mass on PAD/MASK
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
            total = total - (w * F.log_softmax(lk.float(), dim=-1)).sum()
        ntok = ntok + keep.sum()
    return total / ntok.clamp(min=1), ntok


@torch.no_grad()
def bucket_mae(logits_list, targets, fields):
    """Intrinsic ordinal metric: mean |argmax bucket - true bucket| on masked numeric cells.

    Directly measures whether the model learned the *ordering* of a numeric field, independent
    of any downstream label (useful when the numeric field carries no label signal).
    """
    tot = torch.zeros((), device=targets.device)
    n = torch.zeros((), device=targets.device)
    for j, logits in enumerate(logits_list):
        if fields[j].kind not in ORDINAL_KINDS:
            continue
        t = targets[:, :, j].reshape(-1)
        keep = t != IGNORE
        if not bool(keep.any()):
            continue
        pred = logits.reshape(-1, logits.size(-1))[keep].argmax(-1)
        tot = tot + (pred - t[keep]).abs().float().sum()
        n = n + keep.sum()
    return (tot / n.clamp(min=1)), n


# --------------------------------------------------------- ELECTRA-style RTD
def field_marginals(codes: torch.Tensor, key_pad: torch.Tensor, vocabs: list[int]):
    """Empirical per-field value distribution over real cells -> list of cumulative CDFs.

    Sampling replacements from these makes a corruption *plausible* (a value that genuinely
    occurs in that column) rather than uniform noise, which is what makes the detection task
    non-trivial.
    """
    cdfs = []
    real = key_pad.reshape(-1).bool()
    flat = codes.reshape(-1, codes.size(-1))[real]            # (N,F)
    for j, V in enumerate(vocabs):
        cnt = torch.bincount(flat[:, j], minlength=V).float()
        cnt[:N_SPECIAL] = 0.0                                 # never sample PAD/MASK
        p = cnt / cnt.sum().clamp(min=1)
        cdfs.append(torch.cumsum(p, dim=0))
    return cdfs


def apply_rtd_corruption(codes: torch.Tensor, key_pad: torch.Tensor, p_corrupt: float, cdfs):
    """Replace a fraction of real cells with plausible values sampled per field.

    Returns (corrupted_codes, labels, valid) where ``labels`` is 1.0 where the value actually
    changed and 0.0 elsewhere, and ``valid`` marks real (non-pad) cells -- every one of which
    contributes to the loss (unlike MLM, where only masked cells do).
    """
    B, L, F_ = codes.shape
    dev = codes.device
    real = key_pad.unsqueeze(-1).expand(B, L, F_)
    do = (torch.rand(B, L, F_, device=dev) < p_corrupt) & real
    repl = torch.empty_like(codes)
    for j in range(F_):
        u = torch.rand(B, L, device=dev)
        repl[:, :, j] = torch.searchsorted(cdfs[j].to(dev), u.reshape(-1)).reshape(B, L).clamp(
            max=cdfs[j].numel() - 1)
    corrupted = torch.where(do, repl, codes)
    labels = (corrupted != codes).float()      # resampling the original value => label 0 (as in ELECTRA)
    return corrupted, labels, real


def rtd_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor):
    """Binary cross-entropy over ALL real cells. ``logits`` (B,L,F) are per-cell replaced-logits."""
    v = valid.float()
    pos = labels.sum().clamp(min=1)
    neg = v.sum() - pos
    pos_weight = (neg / pos).clamp(1.0, 50.0)                 # corruption rate is low -> reweight
    per = F.binary_cross_entropy_with_logits(
        logits.float(), labels, reduction="none", pos_weight=pos_weight)
    return (per * v).sum() / v.sum().clamp(min=1), v.sum()


@torch.no_grad()
def rtd_accuracy(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor):
    """Detection quality: accuracy on corrupted cells and on clean cells (report both)."""
    pred = (logits > 0).float()
    v = valid.float()
    corr = labels * v
    clean = (1 - labels) * v
    acc_corrupt = ((pred == labels).float() * corr).sum() / corr.sum().clamp(min=1)
    acc_clean = ((pred == labels).float() * clean).sum() / clean.sum().clamp(min=1)
    return acc_corrupt, acc_clean
