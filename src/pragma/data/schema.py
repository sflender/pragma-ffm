"""Per-dataset field schemas for the KVT tokenizer.

A *schema* is just the ordered list of ``(field_name, kind)`` the tokenizer fits over
(``num`` | ``cat`` | ``hash`` | ``cal``; ``dtlog`` is appended by ``--include-dt``). The
tokenizer, encoder, model, dataset, train and probe code are all generic over this list —
only the schema here and the raw→canonical parser (one module per dataset) are
dataset-specific. Adding a dataset = add a field list here + a parser that emits the
canonical parquet (``seq_id, ts, is_fraud, split, amount, <fields…>``).

The canonical parquet must contain a column for every non-derived field name below
(``hour``/``dow`` are derived from ``ts``; ``dt`` from consecutive ``ts`` within a seq).
"""
from __future__ import annotations

# --- TabFormer (IBM) — the original 10 fields. FROZEN: changing this breaks E0–E14 repro.
TABFORMER_FIELDS = [
    ("amount", "num"),
    ("use_chip", "cat"),
    ("mcc", "cat"),
    ("merchant_state", "cat"),
    ("errors", "cat"),
    ("hour", "cal"),          # 0..23  (derived from ts)
    ("dow", "cal"),           # 0..6   (derived from ts)
    ("merchant_name", "hash"),
    ("merchant_city", "hash"),
    ("zip", "hash"),
]

# --- IEEE-CIS (Vesta) — interpretable core fields. card1 is the SEQUENCE entity (not a
# field). Anonymised V/C/D/M engineered columns are left out of the FFM's first cut; the
# relational entities (addr1, P_emaildomain, card2) are shared across card sequences.
IEEE_CIS_FIELDS = [
    ("amount", "num"),          # TransactionAmt
    ("ProductCD", "cat"),       # 5 product codes
    ("card4", "cat"),           # card network (visa/mastercard/amex/discover)
    ("card6", "cat"),           # debit / credit
    ("card2", "cat"),           # ~500 issuer codes
    ("card3", "cat"),           # ~114 codes
    ("card5", "cat"),           # ~119 codes
    ("addr1", "cat"),           # billing region (~332) — also a relational entity
    ("addr2", "cat"),           # billing country (~74)
    ("P_emaildomain", "cat"),   # purchaser email domain (~59)
    ("R_emaildomain", "cat"),   # recipient email domain (~60)
    ("DeviceType", "cat"),      # desktop / mobile (from identity join; mostly NA)
    ("hour", "cal"),            # derived from ts (= reference epoch + TransactionDT)
    ("dow", "cal"),
    ("DeviceInfo", "hash"),     # high-card device string (from identity join)
]

# Synthetic controlled generator (scripts/gen_synth_relational.py). merchant is the shared
# cross-card entity; fraud can be per-card (in the sequence) or relational (merchant state).
SYNTH_FIELDS = [
    ("amount", "num"),
    ("mcc", "cat"),
    ("hour", "cal"),
    ("dow", "cal"),
    ("merchant", "hash"),
]

# TalkingData AdTracking (click fraud). SEQUENCE entity = user (ip+device+os); the shared
# cross-user entity is the IP (fraud farm). Within a user's sequence device/os/ip are constant,
# so the per-click event fields are app/channel (+ hour, dt). No numeric field (clicks have no
# amount); a dummy amount=0 column is emitted for the encoder. is_fraud = is_attributed.
TALKINGDATA_FIELDS = [
    ("app", "cat"),
    ("channel", "cat"),
    ("device", "cat"),
    ("os", "cat"),
    ("hour", "cal"),          # derived from ts
]

CAL_SIZES = {"hour": 24, "dow": 7}

SCHEMAS = {
    "tabformer": TABFORMER_FIELDS,
    "ieee_cis": IEEE_CIS_FIELDS,
    "synth": SYNTH_FIELDS,
    "talkingdata": TALKINGDATA_FIELDS,
}


def get_field_specs(dataset: str) -> list[tuple[str, str]]:
    """Ordered (name, kind) field list for a dataset. Copy so callers can't mutate the registry."""
    if dataset not in SCHEMAS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(SCHEMAS)}")
    return list(SCHEMAS[dataset])
