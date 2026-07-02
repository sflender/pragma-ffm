"""Causal (leakage-safe) feature engineering for the LightGBM fraud baseline.

All history-derived features use only information available strictly *before* the
current transaction (expanding stats are shifted by one within each sequence). This is
the standard strong tabular baseline that a fraud team would actually deploy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ["use_chip", "mcc", "merchant_state", "errors"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to df rows (df must be sorted by seq_id, ts)."""
    g = df.groupby("seq_id", sort=False)
    out = pd.DataFrame(index=df.index)

    # --- raw / calendar ---
    out["amount"] = df["amount"].astype("float32")
    out["log_amount"] = np.sign(df["amount"]) * np.log1p(df["amount"].abs())
    dt = pd.to_datetime(df["ts"].to_numpy(), unit="s")
    out["hour"] = dt.hour.astype("int16")
    out["dow"] = dt.dayofweek.astype("int16")
    out["day"] = dt.day.astype("int16")
    out["month"] = dt.month.astype("int16")
    out["is_online"] = (df["merchant_state"] == "<NA>").astype("int8")

    # --- sequence position / recency ---
    out["pos"] = g.cumcount().astype("int32")
    dt_last = df["ts"].astype("float64") - g["ts"].shift(1)
    out["dt_last_s"] = dt_last.fillna(-1).astype("float32")

    # --- causal expanding amount stats (shifted by 1) ---
    amt = df["amount"].astype("float64")
    cnt = out["pos"].astype("float64")                       # #prior txns (0-based pos)
    csum = g["amount"].cumsum() - amt                        # sum of prior amounts
    csq = g[["amount"]].transform(lambda s: (s * s).cumsum())["amount"] - amt * amt
    prior_mean = np.where(cnt > 0, csum / cnt, 0.0)
    prior_var = np.where(cnt > 0, csq / cnt - prior_mean ** 2, 0.0)
    prior_std = np.sqrt(np.clip(prior_var, 0, None))
    out["amt_prior_mean"] = prior_mean.astype("float32")
    out["amt_prior_std"] = prior_std.astype("float32")
    out["amt_zscore"] = np.where(prior_std > 1e-6, (amt - prior_mean) / prior_std, 0.0).astype("float32")

    # --- categoricals (LightGBM native) ---
    for c in CAT_COLS:
        out[c] = df[c].astype("category")

    return out
