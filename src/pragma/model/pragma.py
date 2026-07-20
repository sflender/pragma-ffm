"""mini-PRAGMA assembly: embeddings -> Event Encoder -> History Encoder, plus MLM heads.

The downstream fraud probe consumes ``record_embeddings`` (per-event outputs of the
History Encoder). MLM reconstructs masked (field, value) tokens from the contextualised
field token fused with the record embedding, giving reconstruction access to both
within-event and cross-event context.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pragma.config import ModelConfig
from pragma.model.encoder import (CrossSequenceEncoder, EventEncoder, FieldValueEmbedder,
                                  HistoryEncoder, MemoryCrossAttention)
from pragma.model.tokenizer import Tokenizer


class MiniPragma(nn.Module):
    def __init__(self, tok: Tokenizer, cfg: ModelConfig):
        super().__init__()
        self.tok = tok
        self.cfg = cfg
        d = cfg.d_model
        self.embedder = FieldValueEmbedder(tok, d, cfg.numeric_mode, cfg.periodic_n_freq,
                                           cfg.use_field_emb)
        self.event = EventEncoder(d, cfg.n_heads, cfg.d_ff, cfg.n_event_layers, cfg.dropout)
        self.history = HistoryEncoder(
            d, cfg.n_heads, cfg.d_ff, cfg.n_history_layers, cfg.dropout, cfg.rope_theta)
        if cfg.use_mem:
            self.mem_norm = nn.LayerNorm(d)
            self.mem = MemoryCrossAttention(d, cfg.n_heads, cfg.d_mem, cfg.dropout)
        if getattr(cfg, "use_xseq", False):
            self.xseq = CrossSequenceEncoder(d, cfg.n_heads, cfg.xseq_layers, cfg.dropout,
                                             count_dim=getattr(cfg, "xseq_count_dim", 0))
        if getattr(cfg, "use_aux_vel", False):
            self.vel_head = nn.Linear(d, cfg.aux_vel_dim)
        self.mlm_norm = nn.LayerNorm(d)
        self.mlm_heads = nn.ModuleList([nn.Linear(d, f.vocab) for f in tok.fields])
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def encode(self, codes, times, mask, amount=None, causal=False, mem=None):
        """Return (record_emb (B,L,d), field_out (B,L,F,d))."""
        tokens = self.embedder(codes, amount)
        evt, field_out = self.event(tokens)
        # positional signal for the History Encoder attention
        pm = self.cfg.pos_mode
        if pm == "index":
            B, L = times.shape
            pos = torch.arange(L, device=times.device, dtype=times.dtype).expand(B, L)
        else:                                   # "time" or "none" (none ignores pos)
            pos = times
        r = self.history(evt, pos, mask, causal, use_rope=(pm != "none"))
        # relational merchant-memory cross-attention (residual, pre-norm)
        if self.cfg.use_mem and mem is not None:
            r = r + self.mem(self.mem_norm(r), mem, mask)
        return r, field_out

    def record_embeddings(self, codes, times, mask, amount=None, causal=False, mem=None):
        return self.encode(codes, times, mask, amount, causal, mem)[0]

    def embed_neighbors(self, nbr_codes, nbr_amount=None):
        """Encode last-K neighbour events with the SHARED embedder+event encoder -> (B,K,d)."""
        tokens = self.embedder(nbr_codes, nbr_amount)       # (B,K,F,d)
        evt, _ = self.event(tokens)                         # (B,K,d)
        return evt

    def apply_xseq(self, target, nbr_codes, nbr_amount, nbr_dt, nbr_mask, count=None):
        """Add the cross-sequence residual to a target-event embedding (B,d). ``count`` is an
        optional precomputed magnitude signal (log-velocity) for the count-aware readout."""
        nbr = self.embed_neighbors(nbr_codes, nbr_amount)
        return target + self.xseq(target, nbr, nbr_dt, nbr_mask, count=count)

    def mlm_logits(self, codes, times, mask, amount=None, mem=None):
        return self.mlm_logits_and_rec(codes, times, mask, amount, mem)[0]

    def mlm_logits_and_rec(self, codes, times, mask, amount=None, mem=None):
        """Return (per-field MLM logits, record embeddings) from a single forward pass, so the
        aligned-SSL aux head can read the same ``r`` the MLM heads are fused with."""
        r, field_out = self.encode(codes, times, mask, amount, mem=mem)
        fused = self.mlm_norm(field_out + r.unsqueeze(2))   # (B,L,F,d)
        logits = [head(fused[:, :, j, :]) for j, head in enumerate(self.mlm_heads)]
        return logits, r
