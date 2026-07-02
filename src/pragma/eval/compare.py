"""Print a side-by-side comparison of all evaluated arms from their JSON artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--files", nargs="*", default=None,
                    help="explicit result json files; default = all *_lgbm.json/probe*.json")
    args = ap.parse_args()

    ad = Path(args.artifacts)
    files = args.files or sorted(
        [str(p) for p in ad.glob("*.json")
         if any(k in p.name for k in ("lgbm", "probe"))])
    rows = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
            if "metrics" in d:
                rows.append((d["arm"], d["metrics"]))
        except Exception as e:  # noqa
            print(f"skip {f}: {e}")

    if not rows:
        print("no result artifacts found")
        return

    print(f"{'arm':<28}{'ROC-AUC':>9}{'PR-AUC':>9}{'R@P0.5':>9}{'R@P0.9':>9}{'n_test':>12}")
    print("-" * 76)
    for arm, m in rows:
        print(f"{arm:<28}{m['roc_auc']:>9.4f}{m['pr_auc']:>9.4f}"
              f"{m['recall@p0.5']:>9.3f}{m['recall@p0.9']:>9.3f}{m['n']:>12,}")


if __name__ == "__main__":
    main()
