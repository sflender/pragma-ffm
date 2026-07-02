"""LightGBM fraud-detection baseline (the incumbent that PRAGMA must beat).

Trains on the train split with early stopping on val, evaluates on test. Reports via
the shared eval harness so numbers are directly comparable to the PRAGMA probe.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pragma.baselines.features import CAT_COLS, build_features
from pragma.eval.metrics import evaluate, print_report


def run(parquet: str, out_json: str, num_rounds: int, seed: int = 0) -> dict:
    t0 = time.time()
    df = pd.read_parquet(parquet, columns=[
        "seq_id", "ts", "amount", "use_chip", "mcc", "merchant_state",
        "errors", "is_fraud", "split"])
    print(f"[lgbm] loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    X = build_features(df)
    y = df["is_fraud"].to_numpy(np.int8)
    split = df["split"].to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"

    pos = int(y[tr].sum()); neg = int((~y[tr].astype(bool)).sum())
    spw = neg / max(1, pos)
    print(f"[lgbm] features={X.shape[1]} train_pos={pos:,} scale_pos_weight={spw:.1f}")

    dtrain = lgb.Dataset(X[tr], label=y[tr], categorical_feature=CAT_COLS, free_raw_data=False)
    dval = lgb.Dataset(X[va], label=y[va], categorical_feature=CAT_COLS, reference=dtrain)
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=128, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, scale_pos_weight=spw,
                  seed=seed, verbosity=-1)
    model = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)])

    scores = model.predict(X[te], num_iteration=model.best_iteration)
    m = evaluate(y[te], scores)
    print_report("LightGBM / test", m)

    imp = sorted(zip(X.columns, model.feature_importance("gain")), key=lambda x: -x[1])
    print("[lgbm] top features:", [f"{k}={v:.0f}" for k, v in imp[:8]])

    result = {"arm": "lightgbm", "best_iteration": model.best_iteration, "metrics": m}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2))
    print(f"[lgbm] wrote {out_json}  (total {time.time()-t0:.1f}s)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/transactions.parquet")
    ap.add_argument("--out", default="artifacts/baseline_lgbm.json")
    ap.add_argument("--num-rounds", type=int, default=500)
    args = ap.parse_args()
    run(args.parquet, args.out, args.num_rounds)


if __name__ == "__main__":
    main()
