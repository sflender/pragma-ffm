"""Key-value-time tokeniser for banking events (PRAGMA-style).

Each transaction (event) is represented as a fixed set of ``(field, value)`` tokens.
Value encoding depends on field type:

  * numerical (``amount``)        -> percentile bucket id (edges fit on TRAIN only)
  * low-cardinality categorical   -> per-field vocabulary id
  * high-cardinality categorical  -> stable hash bucket id (crc32)
  * calendar (``hour``, ``dow``)  -> derived from the timestamp, small vocab

Per field, local value id 0 = <UNK/PAD> and 1 = <MASK>; real values start at 2.
The tokeniser stores a global offset per field so the model can share one big value
embedding table (global id = offset[field] + local id), while MLM heads predict the
local id within a field.

The tokeniser is *fit* on the train split, saved to JSON, and reused everywhere.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PAD, MASK, N_SPECIAL = 0, 1, 2  # reserved local ids present in every field

# (name, type). Order is fixed and defines the field index used everywhere.
FIELD_SPECS = [
    ("amount", "num"),
    ("use_chip", "cat"),
    ("mcc", "cat"),
    ("merchant_state", "cat"),
    ("errors", "cat"),
    ("hour", "cal"),          # 0..23
    ("dow", "cal"),           # 0..6
    ("merchant_name", "hash"),
    ("merchant_city", "hash"),
    ("zip", "hash"),
]
CAL_SIZES = {"hour": 24, "dow": 7}


def _crc(s: str, buckets: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % buckets


@dataclass
class FieldSpec:
    name: str
    kind: str            # num | cat | hash | cal
    vocab: int           # total local vocab size (incl. specials)
    offset: int          # global embedding offset
    cat_map: Optional[dict] = None       # value -> local id (cat only)
    edges: Optional[list] = None         # bucket edges (num only)


class Tokenizer:
    def __init__(self, fields: list[FieldSpec], n_amount_buckets: int, hash_buckets: int):
        self.fields = fields
        self.n_amount_buckets = n_amount_buckets
        self.hash_buckets = hash_buckets
        self.F = len(fields)
        self.V = sum(f.vocab for f in fields)
        self.by_name = {f.name: f for f in fields}

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, df_train: pd.DataFrame, n_amount_buckets: int, hash_buckets: int) -> "Tokenizer":
        fields: list[FieldSpec] = []
        offset = 0
        for name, kind in FIELD_SPECS:
            if kind == "num":
                q = np.linspace(0, 1, n_amount_buckets + 1)
                edges = np.unique(np.quantile(df_train[name].to_numpy(), q)).tolist()
                # local vocab = specials + (len(edges)-1) buckets
                vocab = N_SPECIAL + max(1, len(edges) - 1)
                fields.append(FieldSpec(name, kind, vocab, offset, edges=edges))
            elif kind == "cat":
                vals = df_train[name].astype(str).unique().tolist()
                cat_map = {v: i + N_SPECIAL for i, v in enumerate(sorted(vals))}
                fields.append(FieldSpec(name, kind, N_SPECIAL + len(cat_map), offset, cat_map=cat_map))
            elif kind == "cal":
                fields.append(FieldSpec(name, kind, N_SPECIAL + CAL_SIZES[name], offset))
            elif kind == "hash":
                fields.append(FieldSpec(name, kind, N_SPECIAL + hash_buckets, offset))
            else:
                raise ValueError(kind)
            offset += fields[-1].vocab
        return cls(fields, n_amount_buckets, hash_buckets)

    # ------------------------------------------------------------- encode
    def encode_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Return an int32 array of shape (N, F) of LOCAL value ids (real values >= 2)."""
        n = len(df)
        out = np.zeros((n, self.F), dtype=np.int32)
        # calendar fields need the timestamp
        dt = pd.to_datetime(df["ts"].to_numpy(), unit="s")
        hour = np.asarray(dt.hour)
        dow = np.asarray(dt.dayofweek)
        for j, f in enumerate(self.fields):
            if f.kind == "num":
                edges = np.asarray(f.edges)
                b = np.searchsorted(edges, df[f.name].to_numpy(), side="right") - 1
                b = np.clip(b, 0, f.vocab - N_SPECIAL - 1)
                out[:, j] = b + N_SPECIAL
            elif f.kind == "cat":
                m = f.cat_map
                out[:, j] = df[f.name].astype(str).map(lambda v: m.get(v, PAD)).to_numpy()
            elif f.kind == "cal":
                vals = hour if f.name == "hour" else dow
                out[:, j] = vals + N_SPECIAL
            elif f.kind == "hash":
                col = df[f.name].astype(str).to_numpy()
                out[:, j] = np.fromiter((_crc(s, self.hash_buckets) for s in col),
                                        count=n, dtype=np.int64) + N_SPECIAL
        return out

    # -------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "n_amount_buckets": self.n_amount_buckets,
            "hash_buckets": self.hash_buckets,
            "fields": [
                {"name": f.name, "kind": f.kind, "vocab": f.vocab, "offset": f.offset,
                 "cat_map": f.cat_map, "edges": f.edges}
                for f in self.fields
            ],
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        d = json.loads(Path(path).read_text())
        fields = [FieldSpec(**f) for f in d["fields"]]
        return cls(fields, d["n_amount_buckets"], d["hash_buckets"])
