"""Backfill existing result artifacts (artifacts/*.json) into the local MLflow store.

Parses each arm name into structured params and logs params + metrics as one MLflow run.
Idempotent: skips runs whose run_name already exists, so it's safe to re-run as new
experiments complete.

Run:  python scripts/backfill_mlflow.py
Then: mlflow ui   (browse at http://127.0.0.1:5000)
"""
from __future__ import annotations

import json
from pathlib import Path

from pragma.tracking import clean_metrics, setup

NANO_DEFAULT_L = 96
BIG_DEFAULT_L = 128


def parse_arm(arm: str) -> dict:
    if arm.startswith("lightgbm"):
        return {"arm": arm, "model": "lightgbm", "numeric_mode": "none", "dt": False,
                "pos_mode": "n/a", "seq_len": 0, "seed": 0, "steps": 0, "readout": "n/a"}
    toks = arm.split("_")                     # pragma_<size>_<numeric>_..._<readout>
    if len(toks) < 3:
        return {"arm": arm, "model": "?", "numeric_mode": "?", "dt": False,
                "pos_mode": "n/a", "seq_len": 0, "seed": 0, "steps": 0, "readout": "n/a"}
    size = toks[1]
    rest = toks[3:]
    L = next((int(t[1:]) for t in rest if t.startswith("L") and t[1:].isdigit()), None)
    pos = next((t[3:] for t in rest if t.startswith("pos")), "time")
    seed = next((int(t[1:]) for t in rest if len(t) > 1 and t[0] == "s" and t[1:].isdigit()), 0)
    return {
        "arm": arm, "model": size, "numeric_mode": toks[2],
        "dt": "dt" in rest,
        "pos_mode": pos, "seed": seed,
        "seq_len": L if L else (NANO_DEFAULT_L if size == "nano" else BIG_DEFAULT_L),
        "steps": 6000 if "6k" in rest else 3000,
        "readout": "as-of-date" if "asof" in rest else "bidirectional",
    }


def main():
    mlflow = setup()
    if mlflow is None:
        raise SystemExit("mlflow not installed")
    exp = mlflow.get_experiment_by_name("pragma-ffm")
    client = mlflow.tracking.MlflowClient()
    existing = set()
    if exp:
        for r in client.search_runs([exp.experiment_id], max_results=1000):
            existing.add(r.data.tags.get("mlflow.runName", ""))

    files = sorted(Path("artifacts").glob("*.json"))
    n_new = 0
    for f in files:
        d = json.loads(f.read_text())
        if "metrics" not in d or "arm" not in d:
            continue
        arm = d["arm"]
        if arm in existing:
            continue
        params = parse_arm(arm)
        with mlflow.start_run(run_name=arm):
            mlflow.log_params(params)
            mlflow.log_metrics(clean_metrics(d["metrics"]))
            mlflow.set_tag("source", "backfill")
            if "ckpt" in d:
                mlflow.set_tag("ckpt", d["ckpt"])
        n_new += 1
        print(f"  logged {arm}")
    print(f"[backfill] added {n_new} runs ({len(files)} json files scanned). "
          f"Browse with: mlflow ui  (from {Path.cwd()})")


if __name__ == "__main__":
    main()
