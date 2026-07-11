"""Diagnostic: is the L=256 collapse driven by TIME SPAN (RoPE) or EVENT COUNT (dilution)?

All L=256 windows have the same 256 events but different time spans. We fit the probe on the
L=256 model's w256 embeddings, then bin the *full-256-event* test windows by their time span
and report ROC/AP per bin.
  - collapse concentrated in long-span bins  -> time/RoPE implicated
  - collapse uniform across spans            -> dilution (event count), not RoPE
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pragma.data.dataset import AsOfDateDataset
from pragma.model.tokenizer import Tokenizer
from pragma.train.asof_probe import build_targets
from pragma.train.pretrain import to_device
from pragma.train.probe import load_backbone
from pragma.utils import get_device, seed_everything


@torch.no_grad()
def embed(model, ds, device, want_meta=False, bs=512):
    E, Y, S, N = [], [], [], []
    for b in DataLoader(ds, batch_size=bs):
        b = to_device(b, device)
        r = model.record_embeddings(b["codes"], b["times"], b["mask"], b["amount"], causal=False)
        E.append(r[:, -1].float().cpu().numpy()); Y.append(b["label"].cpu().numpy())
        if want_meta:
            S.append(b["times"][:, -1].cpu().numpy())        # span in days (times[0]=0)
            N.append(b["mask"].sum(1).cpu().numpy())          # #real events in window
    E, Y = np.concatenate(E), np.concatenate(Y)
    return (E, Y, np.concatenate(S), np.concatenate(N)) if want_meta else (E, Y)


device = get_device("auto"); seed_everything(0)
tok = Tokenizer.load("artifacts/tokenizer_dt.json")
model, _, mcfg = load_backbone("artifacts/pretrain_small_bucket_dt_6k_L256.pt", tok, device)
L = mcfg.max_seq_len
rng = np.random.default_rng(0)
tr = build_targets("data/processed_dt", "train", rng, 20000)
te = build_targets("data/processed_dt", "test", rng, 60000)
print(f"targets: train {len(tr):,} test {len(te):,}", flush=True)
Xtr, ytr = embed(model, AsOfDateDataset("data/processed_dt", tr, L), device)
print(f"embedded train: {Xtr.shape} frauds={int(ytr.sum())}", flush=True)
Xte, yte, span, nreal = embed(model, AsOfDateDataset("data/processed_dt", te, L), device, want_meta=True)
print(f"embedded test: {Xte.shape} frauds={int(yte.sum())}", flush=True)

sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
s = clf.predict_proba(sc.transform(Xte))[:, 1]

full = nreal >= L                                            # constant event count (256); only span varies
sf, yf, spf = s[full], yte[full], span[full]
print(f"full-{L}-event test windows: {full.sum():,} (frauds {int(yf.sum()):,})")
edges = np.quantile(spf, [0, .25, .5, .75, 1.0])
print(f"overall (full-256): AP {average_precision_score(yf, sf):.3f} ROC {roc_auc_score(yf, sf):.3f}")
print("--- binned by window time-span (days), event count fixed at 256 ---")
for i in range(4):
    lo, hi = edges[i], edges[i + 1]
    m = (spf >= lo) & (spf <= hi) if i == 3 else (spf >= lo) & (spf < hi)
    if m.sum() < 50 or yf[m].sum() < 5:
        print(f"  span [{lo:6.0f},{hi:6.0f}]d: too few"); continue
    print(f"  span [{lo:6.0f},{hi:6.0f}]d  n={m.sum():>6,} fr={int(yf[m].sum()):>4}: "
          f"AP {average_precision_score(yf[m], sf[m]):.3f}  ROC {roc_auc_score(yf[m], sf[m]):.3f}")
