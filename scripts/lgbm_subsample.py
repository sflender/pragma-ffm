"""LightGBM baseline evaluated on the SAME stratified test subsample as the (a) probe.

Retrains LightGBM (fast) and reports test PR-AUC on exactly the subsample the sliding-window
probe uses (build_targets, seed 0), so FFM-vs-LightGBM is apples-to-apples (same base rate).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from pragma.baselines.features import CAT_COLS, build_features
from pragma.eval.metrics import evaluate, print_report
from pragma.train.asof_probe import build_targets

MAX_NEG = 150000

df = pd.read_parquet("data/processed/transactions.parquet", columns=[
    "seq_id", "ts", "amount", "use_chip", "mcc", "merchant_state", "errors", "is_fraud", "split"])
X = build_features(df)
y = df["is_fraud"].to_numpy(np.int8)
split = df["split"].to_numpy()
tr, va, te = split == "train", split == "val", split == "test"
spw = (~y[tr].astype(bool)).sum() / max(1, int(y[tr].sum()))

dtrain = lgb.Dataset(X[tr], y[tr], categorical_feature=CAT_COLS, free_raw_data=False)
dval = lgb.Dataset(X[va], y[va], categorical_feature=CAT_COLS, reference=dtrain)
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=128,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, scale_pos_weight=spw, seed=0, verbosity=-1)
model = lgb.train(params, dtrain, 800, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
model.save_model("artifacts/lgbm_model.txt", num_iteration=model.best_iteration)
print(f"saved artifacts/lgbm_model.txt (best_iteration={model.best_iteration})")

te_global = np.where(te)[0]                                   # sorted global indices of test rows
scores_te = model.predict(X[te], num_iteration=model.best_iteration)
sub = build_targets("data/processed", "test", np.random.default_rng(0), MAX_NEG)  # same subsample
pos = np.searchsorted(te_global, sub)                         # map subsample -> positions in te
sub_scores, sub_y = scores_te[pos], y[sub]

m = evaluate(sub_y, sub_scores)
print_report("LightGBM / subsample (same as (a))", m)
json.dump({"arm": "lightgbm_swasof", "eval": "subsample", "test_base_rate": float(sub_y.mean()),
           "metrics": m}, open("artifacts/eval_lightgbm_swasof.json", "w"), indent=2)
print("wrote artifacts/eval_lightgbm_swasof.json")
