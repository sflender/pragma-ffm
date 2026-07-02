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
from pragma.model.encoder import EventEncoder, FieldValueEmbedder, HistoryEncoder
from pragma.model.tokenizer import Tokenizer


class MiniPragma(nn.Module):
    def __init__(self, tok: Tokenizer, cfg: ModelConfig):
        super().__init__()
        self.tok = tok
        self.cfg = cfg
        d = cfg.d_model
        self.embedder = FieldValueEmbedder(tok, d, cfg.numeric_mode, cfg.periodic_n_freq)
        self.event = EventEncoder(d, cfg.n_heads, cfg.d_ff, cfg.n_event_layers, cfg.dropout)
        self.history = HistoryEncoder(
            d, cfg.n_heads, cfg.d_ff, cfg.n_history_layers, cfg.dropout, cfg.rope_theta)
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

    def encode(self, codes, times, mask, amount=None, causal=False):
        """Return (record_emb (B,L,d), field_out (B,L,F,d))."""
        tokens = self.embedder(codes, amount)
        evt, field_out = self.event(tokens)
        r = self.history(evt, times, mask, causal)
        return r, field_out

    def record_embeddings(self, codes, times, mask, amount=None, causal=False):
        return self.encode(codes, times, mask, amount, causal)[0]

    def mlm_logits(self, codes, times, mask, amount=None):
        r, field_out = self.encode(codes, times, mask, amount)
        fused = self.mlm_norm(field_out + r.unsqueeze(2))   # (B,L,F,d)
        return [head(fused[:, :, j, :]) for j, head in enumerate(self.mlm_heads)]
