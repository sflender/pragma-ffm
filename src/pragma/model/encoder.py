"""Model building blocks for mini-PRAGMA.

  * FieldValueEmbedder : (field, value) -> vector (field emb + shared value emb).
  * EventEncoder       : set-transformer over an event's field tokens -> [EVT] vector
                         (also returns contextualised field tokens for MLM reconstruction).
  * HistoryEncoder     : bidirectional transformer over the [EVT] sequence with RoPE
                         keyed on continuous event times -> record embedding per event.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pragma.model.tokenizer import Tokenizer


# ------------------------------------------------------------------ embeddings
class FieldValueEmbedder(nn.Module):
    """Local value ids (B,L,F) -> token embeddings (B,L,F,d).

    Token = shared value embedding (global id = per-field offset + local id) + field embedding.
    """

    def __init__(self, tok: Tokenizer, d_model: int):
        super().__init__()
        self.F = tok.F
        offsets = torch.tensor([f.offset for f in tok.fields], dtype=torch.long)
        self.register_buffer("offsets", offsets, persistent=False)
        self.value_emb = nn.Embedding(tok.V, d_model)
        self.field_emb = nn.Embedding(tok.F, d_model)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:  # (B,L,F) long
        gids = codes + self.offsets                          # broadcast over F
        tok = self.value_emb(gids)                           # (B,L,F,d)
        fidx = torch.arange(self.F, device=codes.device)
        return tok + self.field_emb(fidx)                    # + (F,d)


# --------------------------------------------------------------- event encoder
class EventEncoder(nn.Module):
    """Transformer over the F field tokens of each event, plus a learned [EVT] token.

    Returns the [EVT] output (event vector) and the contextualised field tokens.
    Fields are an unordered set -> no positional encoding here.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float):
        super().__init__()
        self.evt = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.evt, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor):                 # (B,L,F,d)
        B, L, Fn, d = tokens.shape
        x = tokens.reshape(B * L, Fn, d)
        evt = self.evt.expand(B * L, 1, d)
        x = torch.cat([evt, x], dim=1)                       # (B*L, 1+F, d)
        h = self.enc(x)                                      # no mask: all fields present
        evt_out = h[:, 0].reshape(B, L, d)                   # (B,L,d)
        field_out = h[:, 1:].reshape(B, L, Fn, d)            # (B,L,F,d)
        return evt_out, field_out


# ------------------------------------------------------------------- rope attn
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B,H,L,dh); cos/sin: (B,1,L,dh)
    return x * cos + _rotate_half(x) * sin


class RoPEAttention(nn.Module):
    """Multi-head self-attention with rotary embeddings on *continuous* event times."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, theta: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = dropout
        inv_freq = theta ** (-torch.arange(0, self.dh, 2).float() / self.dh)
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (dh/2,)

    def _cos_sin(self, times: torch.Tensor):                 # times (B,L)
        ang = times[..., None] * self.inv_freq               # (B,L,dh/2)
        ang = ang.repeat_interleave(2, dim=-1)               # (B,L,dh)
        return ang.cos()[:, None], ang.sin()[:, None]        # (B,1,L,dh)

    def forward(self, x: torch.Tensor, times: torch.Tensor, key_pad: torch.Tensor,
                causal: bool = False):
        # x:(B,L,d)  times:(B,L)  key_pad:(B,L) True=real
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # (B,H,L,dh)
        cos, sin = self._cos_sin(times)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        # additive mask: -1e4 fully masks (scores are O(10)); finite avoids MPS NaNs.
        attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~key_pad[:, None, None, :], -1e4)
        if causal:
            # position i may attend only to keys j <= i (no future events)
            i = torch.arange(L, device=x.device)
            tri = torch.where(i[None, :] <= i[:, None], 0.0, -1e4).to(q.dtype)
            attn_mask = attn_mask + tri[None, None]          # (B,1,L,L)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, d)
        return self.proj(out)


class HistoryLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, theta):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, dropout, theta)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, times, key_pad, causal=False):
        x = x + self.drop(self.attn(self.n1(x), times, key_pad, causal))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class HistoryEncoder(nn.Module):
    """Bidirectional (or causal) transformer over event vectors with time-aware RoPE."""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout, theta):
        super().__init__()
        self.layers = nn.ModuleList(
            [HistoryLayer(d_model, n_heads, d_ff, dropout, theta) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, times, key_pad, causal=False):
        for lyr in self.layers:
            x = lyr(x, times, key_pad, causal)
        return self.norm(x)
