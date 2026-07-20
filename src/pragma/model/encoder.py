"""Model building blocks for mini-PRAGMA.

  * FieldValueEmbedder : (field, value) -> vector (field emb + shared value emb).
  * EventEncoder       : set-transformer over an event's field tokens -> [EVT] vector
                         (also returns contextualised field tokens for MLM reconstruction).
  * HistoryEncoder     : bidirectional transformer over the [EVT] sequence with RoPE
                         keyed on continuous event times -> record embedding per event.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pragma.model.tokenizer import MASK, Tokenizer


# ---------------------------------------------------------- numeric encoders
class PLEEmbedder(nn.Module):
    """Piecewise-Linear Encoding (Gorishniy et al. 2022).

    A value x maps to a T-dim ramp vector: bins fully below x -> 1, the active bin ->
    the fractional position within it, bins above -> 0 (e.g. [1,1,1,0.42,0,0,0,0]), then
    a linear map to d_model. Reuses the tokenizer's quantile bin edges, so it is scale-
    robust and, unlike hard bucketing, preserves within-bin magnitude & ordering.
    """

    def __init__(self, edges, d_model):
        super().__init__()
        e = torch.tensor(edges, dtype=torch.float32)
        self.register_buffer("lo", e[:-1], persistent=False)
        self.register_buffer("hi", e[1:], persistent=False)
        self.T = len(edges) - 1
        self.lin = nn.Linear(self.T, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x (B,L) -> (B,L,d)
        denom = (self.hi - self.lo).clamp(min=1e-6)
        e = ((x[..., None] - self.lo) / denom).clamp(0.0, 1.0)
        return self.lin(e)


class PeriodicEmbedder(nn.Module):
    """Periodic (Fourier-feature) encoding: x -> [sin(2*pi*c*x), cos(2*pi*c*x)] with
    learned frequencies c, then a linear map. Amount is signed-log normalised first to
    tame the heavy tail before the sinusoids.
    """

    def __init__(self, d_model, n_freq: int = 16, sigma: float = 1.0):
        super().__init__()
        self.coef = nn.Parameter(torch.randn(n_freq) * sigma)
        self.lin = nn.Linear(2 * n_freq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x (B,L) -> (B,L,d)
        xn = torch.sign(x) * torch.log1p(x.abs()) / 10.0     # ~[-1,1] for typical amounts
        v = 2 * math.pi * xn[..., None] * self.coef          # (B,L,n_freq)
        e = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B,L,2*n_freq)
        return self.lin(e)


# ------------------------------------------------------------------ embeddings
class FieldValueEmbedder(nn.Module):
    """Local value ids (B,L,F) -> token embeddings (B,L,F,d).

    Token = shared value embedding (global id = per-field offset + local id) + field
    embedding. For numeric fields, ``numeric_mode`` in {ple, periodic} replaces the
    bucket-lookup embedding with a continuous encoding of the raw value; masked numeric
    cells (code == MASK) use a learned mask vector so no value leaks.
    """

    def __init__(self, tok: Tokenizer, d_model: int, numeric_mode: str = "bucket",
                 periodic_n_freq: int = 16, use_field_emb: bool = True):
        super().__init__()
        self.F = tok.F
        self.numeric_mode = numeric_mode
        self.use_field_emb = use_field_emb
        offsets = torch.tensor([f.offset for f in tok.fields], dtype=torch.long)
        self.register_buffer("offsets", offsets, persistent=False)
        self.value_emb = nn.Embedding(tok.V, d_model)
        self.field_emb = nn.Embedding(tok.F, d_model)

        self.num_idx = [i for i, f in enumerate(tok.fields) if f.kind == "num"]
        if numeric_mode != "bucket" and self.num_idx:
            self.num_embedders = nn.ModuleDict()
            self.num_mask = nn.ParameterDict()
            for i in self.num_idx:
                f = tok.fields[i]
                if numeric_mode == "ple":
                    self.num_embedders[str(i)] = PLEEmbedder(f.edges, d_model)
                elif numeric_mode == "periodic":
                    self.num_embedders[str(i)] = PeriodicEmbedder(d_model, periodic_n_freq)
                else:
                    raise ValueError(f"unknown numeric_mode {numeric_mode!r}")
                p = nn.Parameter(torch.zeros(d_model)); nn.init.normal_(p, std=0.02)
                self.num_mask[str(i)] = p

    def forward(self, codes: torch.Tensor, numeric: torch.Tensor | None = None) -> torch.Tensor:
        gids = codes + self.offsets                          # (B,L,F)
        tok = self.value_emb(gids)                           # (B,L,F,d)
        if self.numeric_mode != "bucket" and self.num_idx:
            assert numeric is not None, "numeric values required for ple/periodic mode"
            tok = tok.clone()
            for i in self.num_idx:                           # single numeric field (amount)
                emb = self.num_embedders[str(i)](numeric)    # (B,L,d)
                masked = (codes[:, :, i] == MASK)[..., None]
                tok[:, :, i, :] = torch.where(masked, self.num_mask[str(i)], emb)
        fidx = torch.arange(self.F, device=codes.device)
        return tok + self.field_emb(fidx) if self.use_field_emb else tok


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
    # x: (B,H,L,dh); cos/sin: (B,1,L,dh). Match x's dtype so bf16 autocast doesn't get
    # promoted back to fp32 here (which would make q/k fp32 while v stays bf16 and break SDPA).
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
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
                causal: bool = False, use_rope: bool = True):
        # x:(B,L,d)  times:(B,L)  key_pad:(B,L) True=real
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # (B,H,L,dh)
        if use_rope:
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


class MemoryCrossAttention(nn.Module):
    """Cross-sequence relational block: each event (query) attends over the window's
    per-event MERCHANT-memory tokens (keys/values). The memory vector for an event is a
    precomputed, causal, cross-card summary of its merchant's recent activity, so this
    injects "what's happening at this merchant across all cards" into the record embedding
    — signal a per-sequence encoder cannot see. Applied once after the History Encoder.
    """

    def __init__(self, d_model: int, n_heads: int, d_mem: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        # project the raw memory vector -> a d_model memory token per event
        self.mem_proj = nn.Sequential(nn.Linear(d_mem, d_model), nn.GELU(),
                                      nn.LayerNorm(d_model))
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = dropout

    def forward(self, x, mem, key_pad):
        # x:(B,L,d) events  mem:(B,L,d_mem)  key_pad:(B,L) True=real
        B, L, d = x.shape
        m = self.mem_proj(mem)                                  # (B,L,d) memory tokens
        q = self.q(x).reshape(B, L, self.h, self.dh).transpose(1, 2)
        kv = self.kv(m).reshape(B, L, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                     # (B,H,L,dh)
        attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~key_pad[:, None, None, :], -1e4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.drop if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, L, d))


class CrossSequenceEncoder(nn.Module):
    """The "third transformer": the target event cross-attends over its ENTITY's last-K raw
    prior events (across all cards), encoded by the shared Event Encoder.

    Where ``MemoryCrossAttention`` reads a rank-1, hand-designed summary vector per event
    (popularity, prior fraud rate, windowed velocity...), this sees the actual K neighbour
    events and lets the model *learn* the cross-card pattern (e.g. a velocity burst) rather
    than trusting a precomputed feature. A learned recency (log-dt) signal is added to each
    neighbour, an optional shallow transformer mixes the K neighbours, and the target queries
    them. Returns a residual to add to the target's record embedding. All masks are finite
    (-1e4) so all-padded rows (targets with no prior neighbour) stay NaN-free; their output is
    zeroed explicitly.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float, count_dim: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.drop = dropout
        self.dt_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # count-aware path: an always-on readout of a precomputed magnitude signal (e.g. the
        # entity's windowed velocity / log-count) that the bounded-K raw-neighbour window cannot
        # represent. Pattern via attention (below) + magnitude via this scalar readout, in one module.
        self.count_dim = count_dim
        if count_dim > 0:
            self.count_proj = nn.Sequential(nn.Linear(count_dim, d_model), nn.GELU(),
                                            nn.LayerNorm(d_model))
        # shallow self-attention stack over the K neighbours (finite-mask SDPA, NaN-safe)
        self.layers = nn.ModuleList()
        for _ in range(max(0, n_layers)):
            self.layers.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d_model), "qkv": nn.Linear(d_model, 3 * d_model),
                "op": nn.Linear(d_model, d_model), "n2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                    nn.Linear(4 * d_model, d_model))}))
        # target -> neighbours cross-attention
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _self_attn(self, x, amask):                          # x:(B,K,d) amask:(B,1,1,K)
        B, K, d = x.shape
        for ly in self.layers:
            h = ly["n1"](x)
            qkv = ly["qkv"](h).reshape(B, K, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]                 # (B,H,K,dh)
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=amask, dropout_p=self.drop if self.training else 0.0)
            x = x + ly["op"](o.transpose(1, 2).reshape(B, K, d))
            x = x + ly["ff"](ly["n2"](x))
        return x

    def forward(self, target, nbr, dt, nbr_mask, count=None):
        # target:(B,d) record emb of target event; nbr:(B,K,d) encoded neighbour events;
        # dt:(B,K) seconds from neighbour to target (>=0); nbr_mask:(B,K) True=real;
        # count:(B,count_dim) precomputed magnitude signal (log-velocity), always-on.
        B, K, d = nbr.shape
        logdt = torch.log1p(dt.clamp(min=0.0))[..., None].to(nbr.dtype)   # (B,K,1)
        x = nbr + self.dt_mlp(logdt)
        amask = torch.zeros(B, 1, 1, K, device=x.device, dtype=x.dtype)
        amask = amask.masked_fill(~nbr_mask[:, None, None, :], -1e4)
        x = self._self_attn(x, amask)
        q = self.q(target).reshape(B, self.h, 1, self.dh)
        kv = self.kv(x).reshape(B, K, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                              # (B,H,K,dh)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=amask, dropout_p=self.drop if self.training else 0.0)
        out = self.proj(out.reshape(B, d))
        has = nbr_mask.any(dim=1, keepdim=True)                         # (B,1) targets w/ >=1 nbr
        pattern = torch.where(has, self.norm(out), torch.zeros_like(out))   # attention path (zeroed if no nbr)
        if self.count_dim > 0 and count is not None:                    # + always-on magnitude path
            return pattern + self.count_proj(count.to(pattern.dtype))
        return pattern


class HistoryLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, theta):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, dropout, theta)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, times, key_pad, causal=False, use_rope=True):
        x = x + self.drop(self.attn(self.n1(x), times, key_pad, causal, use_rope))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class HistoryEncoder(nn.Module):
    """Bidirectional (or causal) transformer over event vectors with time-aware RoPE."""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout, theta):
        super().__init__()
        self.layers = nn.ModuleList(
            [HistoryLayer(d_model, n_heads, d_ff, dropout, theta) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, times, key_pad, causal=False, use_rope=True):
        for lyr in self.layers:
            x = lyr(x, times, key_pad, causal, use_rope)
        return self.norm(x)
