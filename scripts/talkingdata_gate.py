"""TalkingData relational gate: does raw-neighbour attention (xseq) beat an engineered IP-velocity
summary — the win we could not get on Elliptic (where a rich pre-baked summary already won)?

Staged so the cheap, decisive signal prints first:
  GATE 1 (LightGBM): local vs local+agg. If +agg >> local, the IP-velocity (relational) signal is
                     real and this dataset is worth the full run.
  GATE 2 (torch, same embedding base): local / +agg / +meanpool / +xseq / +agg+xseq. The key
                     contrast is +xseq vs +agg (learned raw-neighbour attention vs the summary).

Positive class = is_attributed (rare). Metric: PR-AUC + ROC-AUC on the temporal test split.

Run: python scripts/talkingdata_gate.py --data data/td/td.npz --device cpu
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

from pragma.model.encoder import CrossSequenceEncoder


def metrics(y, s):
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}


class TDModel(nn.Module):
    def __init__(self, vocab, d_agg, d=96, arm="local"):
        super().__init__()
        self.arm = arm
        self.emb = nn.ModuleList([nn.Embedding(int(v) + 1, d) for v in vocab])   # app,dev,os,ch,hour
        self.enc = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        if "agg" in arm:
            self.agg_proj = nn.Sequential(nn.Linear(d_agg, d), nn.GELU(), nn.LayerNorm(d))
        if arm in ("local+meanpool", "local+xseq", "local+agg+xseq"):
            pass  # neighbours reuse self.emb[:4] (app,dev,os,ch)
        if "xseq" in arm:
            self.xseq = CrossSequenceEncoder(d, n_heads=4, n_layers=1, dropout=0.1)
        self.head = nn.Linear(d, 1)

    def _emb_local(self, codes):                       # codes (B,5)
        return sum(self.emb[j](codes[:, j]) for j in range(len(self.emb)))

    def _emb_nbr(self, nbr):                            # nbr (B,K,4)
        return sum(self.emb[j](nbr[:, :, j]) for j in range(4))

    def forward(self, codes, agg=None, nbr=None, dt=None, mask=None):
        r = self.enc(self._emb_local(codes))
        if self.arm == "local+agg":
            r = r + self.agg_proj(agg)
        elif self.arm == "local+meanpool":
            ne = self._emb_nbr(nbr); m = mask[..., None].float()
            r = r + (ne * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.arm == "local+xseq":
            r = r + self.xseq(r, self._emb_nbr(nbr), dt, mask)
        elif self.arm == "local+agg+xseq":
            r = r + self.agg_proj(agg) + self.xseq(r, self._emb_nbr(nbr), dt, mask)
        return self.head(r).squeeze(-1)


def run_torch(arm, Z, tr_rows, te_rows, device, epochs, bs, lr, max_neg, d=96):
    vocab = Z["vocab"]; y = Z["y"].astype(np.float32)
    codes_all = torch.tensor(Z["local"], device=device)
    agg_all = torch.tensor(Z["agg"], device=device)
    has_nbr = "nbr" in Z
    if has_nbr:
        nbr_all = torch.tensor(Z["nbr"].astype(np.int64), device=device)
        dt_all = torch.tensor(Z["nbr_dt"], device=device)
        msk_all = torch.tensor(Z["nbr_mask"], device=device)
    model = TDModel(vocab, agg_all.shape[1], d, arm).to(device)

    # class-balanced train subsample (all pos + capped neg) to keep the gate fast
    ytr = y[tr_rows]; pos = tr_rows[ytr == 1]; neg = tr_rows[ytr == 0]
    rng = np.random.default_rng(0)
    if len(neg) > max_neg: neg = rng.choice(neg, max_neg, replace=False)
    use = np.concatenate([pos, neg]); rng.shuffle(use)
    pw = torch.tensor([len(neg) / max(1, len(pos))], device=device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def gather(rows):
        r = torch.tensor(rows, device=device)
        out = [codes_all[r], agg_all[r]]
        out += [nbr_all[r], dt_all[r], msk_all[r]] if has_nbr else [None, None, None]
        return out

    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(use))
        for i in range(0, len(perm), bs):
            rows = use[perm[i:i + bs]]
            c, a, nb, dt, mk = gather(rows)
            yv = torch.tensor(y[rows], device=device)
            loss = lossfn(model(c, a, nb, dt, mk), yv)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval(); scores = []
    with torch.no_grad():
        for i in range(0, len(te_rows), 8192):
            rows = te_rows[i:i + 8192]
            c, a, nb, dt, mk = gather(rows)
            scores.append(torch.sigmoid(model(c, a, nb, dt, mk)).cpu().numpy())
    return metrics(y[te_rows], np.concatenate(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-neg", type=int, default=600000)
    ap.add_argument("--out", default="artifacts/talkingdata_results.json")
    args = ap.parse_args()

    t0 = time.time(); device = torch.device(args.device)
    Z = {k: np.load(args.data)[k] for k in np.load(args.data).files}
    split, y = Z["split"], Z["y"]
    tr_rows, te_rows = np.where(split == 0)[0], np.where(split == 1)[0]
    res = {}

    # ---- GATE 1: LightGBM local vs local+agg ----
    import lightgbm as lgb
    Xl = Z["local"].astype(np.float32); Xa = np.hstack([Xl, Z["agg"]])
    cat = [0, 1, 2, 3, 4]
    for name, X, cf in [("lgbm_local", Xl, cat), ("lgbm_localagg", Xa, cat)]:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                               class_weight="balanced", verbose=-1)
        m.fit(X[tr_rows], y[tr_rows], categorical_feature=cf)
        res[name] = metrics(y[te_rows], m.predict_proba(X[te_rows])[:, 1])
        print(f"[td] {name:16s} PR {res[name]['pr_auc']:.3f} ROC {res[name]['roc_auc']:.3f}", flush=True)
    print(f"[td] GATE1 delta(+agg) = {res['lgbm_localagg']['pr_auc'] - res['lgbm_local']['pr_auc']:+.3f} "
          f"(>0 => IP relational signal is real)", flush=True)

    # ---- GATE 2: torch arms ----
    arms = ["local", "local+agg", "local+meanpool", "local+xseq", "local+agg+xseq"]
    if "nbr" not in Z: arms = ["local", "local+agg"]
    for arm in arms:
        res[arm] = run_torch(arm, Z, tr_rows, te_rows, device, args.epochs, args.batch_size,
                             args.lr, args.max_neg)
        print(f"[td] {arm:16s} PR {res[arm]['pr_auc']:.3f} ROC {res[arm]['roc_auc']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": "talkingdata", "n_test": int(len(te_rows)),
                                          "test_base": float(y[te_rows].mean()), "arms": res}, indent=2))
    print(f"[td] wrote {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
