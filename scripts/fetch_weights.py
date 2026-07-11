"""Fetch pretrained checkpoints from the GitHub release into artifacts/.

The .pt weights are too large for git (see .gitignore); they're published as assets on a
tagged GitHub release. The tokenizers are committed to the repo, so after cloning you only
need the weights:

    python scripts/fetch_weights.py            # all three (nano, mini, small)
    python scripts/fetch_weights.py --only small

Then, e.g.:
    python -m pragma.train.asof_probe --ckpt artifacts/pretrain_small_bucket_dt_6k.pt \
        --data-dir data/processed_dt --tokenizer artifacts/tokenizer_dt.json
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "sflender/pragma-ffm"
TAG = "v0.1"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

# name -> release asset filename (each ckpt embeds its own model_cfg/preset/step)
CKPTS = {
    "nano":  "pretrain_nano_bucket_dt.pt",
    "mini":  "pretrain_mini_bucket_dt.pt",
    "small": "pretrain_small_bucket_dt_6k.pt",
}
EXTRA = ["lgbm_model.txt"]              # LightGBM baseline booster (always fetched)
ART = Path("artifacts")


def _download(url: str, dest: Path):
    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            sys.stdout.write(f"\r  {dest.name}: {pct:3d}%")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, hook)
    sys.stdout.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(CKPTS), help="fetch a single model (default: all)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    names = [args.only] if args.only else list(CKPTS)
    for n in names:
        fname = CKPTS[n]
        dest = ART / fname
        if dest.exists() and not args.force:
            print(f"  {fname}: present ({dest.stat().st_size // 1_000_000} MB), skipping")
            continue
        print(f"fetching {n} <- {BASE}/{fname}")
        _download(f"{BASE}/{fname}", dest)
    if not args.only:                              # baseline booster comes with the full pull
        for fname in EXTRA:
            dest = ART / fname
            if dest.exists() and not args.force:
                print(f"  {fname}: present, skipping")
                continue
            print(f"fetching {fname} <- {BASE}/{fname}")
            _download(f"{BASE}/{fname}", dest)
    print(f"done -> {ART}/ (tokenizers are already in the repo: artifacts/tokenizer*.json)")


if __name__ == "__main__":
    main()
