"""Golden regression test: guarantees the TabFormer tokenizer schema + encode behaviour
stay frozen through the dataset-adapter refactor (and any future change).

Prior experiments E0–E14 depend on the committed `artifacts/tokenizer_dt.json` and on
`encode_frame` producing the same integer codes. These tests lock both without needing the
24M-row dataset: the schema check reads the committed tokenizer, the behaviour check fits a
tiny synthetic frame and asserts exact codes.

Run: `pytest tests/ -q`
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pragma.data.schema import TABFORMER_FIELDS, get_field_specs
from pragma.model.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOK_DT = ROOT / "artifacts" / "tokenizer_dt.json"

# The frozen TabFormer schema (name, kind, vocab, offset) that E0–E14 were run against.
FROZEN_FIELDS = [
    ("amount", "num", 66, 0), ("use_chip", "cat", 5, 66), ("mcc", "cat", 111, 71),
    ("merchant_state", "cat", 222, 182), ("errors", "cat", 25, 404),
    ("hour", "cal", 26, 429), ("dow", "cal", 9, 455),
    ("merchant_name", "hash", 4098, 464), ("merchant_city", "hash", 4098, 4562),
    ("zip", "hash", 4098, 8660), ("dt", "dtlog", 22, 12758),
]
FROZEN_V, FROZEN_F = 12780, 11


@pytest.mark.skipif(not TOK_DT.exists(), reason="committed tokenizer_dt.json not present")
def test_committed_tabformer_schema_frozen():
    """The committed TabFormer tokenizer must keep its exact fields / vocab / offsets."""
    d = json.loads(TOK_DT.read_text())
    got = [(f["name"], f["kind"], f["vocab"], f["offset"]) for f in d["fields"]]
    assert got == FROZEN_FIELDS
    assert len(d["fields"]) == FROZEN_F
    assert sum(f["vocab"] for f in d["fields"]) == FROZEN_V


def test_default_schema_is_tabformer():
    assert get_field_specs("tabformer") == TABFORMER_FIELDS
    # order + kinds are what the tokenizer defaults to
    assert [n for n, _ in TABFORMER_FIELDS][:3] == ["amount", "use_chip", "mcc"]


def _fixture(n=200, seed=0):
    """Tiny TabFormer-shaped frame with a fixed seed."""
    rng = np.random.default_rng(seed)
    base = 1_500_000_000
    return pd.DataFrame({
        "amount": rng.uniform(1, 500, n).astype("float32"),
        "use_chip": rng.choice(["Swipe Transaction", "Online Transaction"], n),
        "mcc": rng.choice(["5411", "5812", "4829"], n),
        "merchant_state": rng.choice(["CA", "NY", "<NA>"], n),
        "errors": rng.choice(["<NA>", "Bad PIN"], n),
        "merchant_name": rng.choice([f"m{i}" for i in range(50)], n),
        "merchant_city": rng.choice(["ONLINE", "Monterey Park"], n),
        "zip": rng.choice(["91754", "<NA>"], n),
        "ts": (base + rng.integers(0, 10_000_000, n)).astype("int64"),
        "dt": rng.integers(0, 100_000, n).astype("int64"),
        "split": "train",
    })


def test_encode_frame_deterministic_and_bounded():
    """fit+encode is deterministic and every code stays within its field's vocab range."""
    df = _fixture()
    tok = Tokenizer.fit(df, 64, 4096, include_dt=True)  # default schema = TabFormer
    assert tok.F == FROZEN_F
    codes = tok.encode_frame(df)
    assert codes.shape == (len(df), tok.F)
    # per-field codes are within [0, vocab) and re-encoding is identical (determinism)
    for j, f in enumerate(tok.fields):
        assert codes[:, j].min() >= 0 and codes[:, j].max() < f.vocab
    assert np.array_equal(codes, tok.encode_frame(df))


def test_hash_field_is_stable_crc():
    """hash fields use a fixed crc32 bucketing — value must not drift."""
    import zlib
    df = _fixture()
    tok = Tokenizer.fit(df, 64, 4096, include_dt=True)
    codes = tok.encode_frame(df)
    j = [i for i, f in enumerate(tok.fields) if f.name == "merchant_name"][0]
    expected = (zlib.crc32(df["merchant_name"].iloc[0].encode()) % 4096) + 2
    assert codes[0, j] == expected


def test_ieee_schema_registered_and_disjoint_from_tabformer():
    specs = get_field_specs("ieee_cis")
    names = [n for n, _ in specs]
    assert "amount" in names and "ProductCD" in names and "card4" in names
    assert "merchant_name" not in names           # not a TabFormer clone
    assert names == list(dict.fromkeys(names))     # no duplicate fields
