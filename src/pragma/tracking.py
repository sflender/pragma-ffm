"""Local MLflow setup (file backend under ./mlruns). Fail-safe: never breaks training.

Used both by the backfill script and (opt-in) by live logging in training/eval.
"""
from __future__ import annotations

from pathlib import Path

EXPERIMENT = "pragma-ffm"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup():
    """Point MLflow at the local file store and select the experiment. Returns the mlflow
    module, or None if mlflow isn't installed (so callers degrade gracefully)."""
    try:
        import mlflow
    except Exception:
        return None
    mlflow.set_tracking_uri(f"sqlite:///{repo_root()}/mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


# metric names must avoid characters like '@' / '.' for MLflow
_METRIC_MAP = {
    "roc_auc": "roc_auc", "pr_auc": "pr_auc",
    "recall@p0.5": "recall_at_p50", "recall@p0.9": "recall_at_p90",
    "n_pos": "n_pos", "n": "n_test",
}


def clean_metrics(m: dict) -> dict:
    return {_METRIC_MAP.get(k, k.replace("@", "_at_").replace(".", "_")): float(v)
            for k, v in m.items() if isinstance(v, (int, float))}
