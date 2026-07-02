"""Shared evaluation metrics + reporting for both arms (LightGBM and PRAGMA probe).

Fraud is extremely imbalanced, so PR-AUC (average precision) is the headline metric;
ROC-AUC is reported alongside. Also reports recall at a few precision thresholds.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def recall_at_precision(y_true, y_score, target_precision: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    ok = p >= target_precision
    return float(r[ok].max()) if ok.any() else 0.0


def evaluate(y_true, y_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    return {
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall@p0.5": recall_at_precision(y_true, y_score, 0.5),
        "recall@p0.9": recall_at_precision(y_true, y_score, 0.9),
    }


def print_report(name: str, m: dict) -> None:
    print(f"[{name}] n={m['n']:,} pos={m['n_pos']:,}  "
          f"ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"R@P0.5={m['recall@p0.5']:.3f}  R@P0.9={m['recall@p0.9']:.3f}")
