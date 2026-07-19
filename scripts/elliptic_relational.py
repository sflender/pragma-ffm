"""External-validity test on Elliptic: does attending over RAW neighbours beat a precomputed
neighbour summary — the §8 synthetic result, on real relational (AML) fraud?

Arms (same local encoder MLP; illicit = positive class):
  lgbm_local     LightGBM on 93 local features                         (tabular per-node reference)
  lgbm_localagg  LightGBM on 93 local + 72 aggregated features          (tabular +summary reference)
  local          MLP(local)                                            per-node FFM analog (blind)
  local+agg      MLP(local) (+) proj(agg)                              rank-1 precomputed summary
  local+meanpool MLP(local) (+) mean_k MLP(neighbour_local)            raw neighbours, no attention
  local+xseq     MLP(local) (+) CrossSequenceEncoder(raw neighbours)   the cross-sequence encoder

The clean contrast is **local+xseq vs local+agg**: learned attention over raw neighbours vs the
hand-engineered neighbour aggregates Elliptic ships with. Reports illicit PR-AUC, ROC-AUC, and the
illicit-class F1@0.5 (the metric the Elliptic literature uses).

Run: python scripts/elliptic_relational.py --data data/elliptic/elliptic.npz --device cpu
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from pragma.model.encoder import CrossSequenceEncoder


def zscore(x, tr):
    mu, sd = x[tr].mean(0), x[tr].std(0); sd[sd < 1e-6] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def metrics(y, s):
    f1 = max(f1_score(y, (s >= t).astype(int)) for t in np.linspace(0.05, 0.95, 19))
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "illicit_f1": float(f1)}


class NodeModel(nn.Module):
    def __init__(self, d_local, d_agg, d=128, arm="local", k=16):
        super().__init__()
        self.arm = arm
        self.enc = nn.Sequential(nn.Linear(d_local, d), nn.GELU(), nn.LayerNorm(d),
                                 nn.Linear(d, d), nn.GELU())
        if arm in ("local+agg", "local+agg+xseq"):
            self.agg_proj = nn.Sequential(nn.Linear(d_agg, d), nn.GELU(), nn.LayerNorm(d))
        if arm in ("local+meanpool", "local+xseq", "local+agg+xseq"):
            self.nbr_enc = nn.Sequential(nn.Linear(d_local, d), nn.GELU(), nn.LayerNorm(d))
        if arm in ("local+xseq", "local+agg+xseq"):
            self.xseq = CrossSequenceEncoder(d, n_heads=4, n_layers=1, dropout=0.1)
        self.head = nn.Linear(d, 1)

    def forward(self, x_local, x_agg=None, nbr_x=None, nbr_dt=None, nbr_mask=None):
        r = self.enc(x_local)                                   # (B,d)
        if self.arm == "local+agg":
            r = r + self.agg_proj(x_agg)
        elif self.arm == "local+meanpool":
            ne = self.nbr_enc(nbr_x)                            # (B,K,d)
            m = nbr_mask[..., None].float()
            r = r + (ne * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.arm == "local+xseq":
            ne = self.nbr_enc(nbr_x)                            # (B,K,d)
            r = r + self.xseq(r, ne, nbr_dt, nbr_mask)
        elif self.arm == "local+agg+xseq":                     # engineered summary AND raw attention
            r = r + self.agg_proj(x_agg)
            ne = self.nbr_enc(nbr_x)
            r = r + self.xseq(r, ne, nbr_dt, nbr_mask)
        return self.head(r).squeeze(-1)


def run_torch(arm, D, device, epochs, bs, lr, d=128):
    xl, xa, yb = D["xl"], D["xa"], D["y"]
    xl_all = torch.tensor(D["xl_all"], device=device)          # (Nall, d_local) for neighbour gather
    step_all = torch.tensor(D["step_all"].astype(np.float32), device=device)  # (Nall,)
    nbr_idx = D["nbr_idx"]
    model = NodeModel(xl.shape[1], xa.shape[1], d, arm, nbr_idx.shape[1]).to(device)
    pw = torch.tensor([(yb["train"] == 0).sum() / max(1, (yb["train"] == 1).sum())], device=device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def batch_tensors(rows):
        rows_t = torch.tensor(rows, device=device)
        bx = torch.tensor(xl[rows], device=device)
        ba = torch.tensor(xa[rows], device=device)
        ni = torch.tensor(nbr_idx[rows], device=device).long()  # (B,K), -1 pad
        nm = ni >= 0                                            # (B,K) True=real neighbour
        safe = ni.clamp(min=0)
        nbx = xl_all[safe] * nm[..., None]                      # (B,K,d_local), padded->0
        # dt = |timestep(node) - timestep(neighbour)| as a weak recency signal
        dt = (step_all[rows_t][:, None] - step_all[safe]).abs() * nm
        return bx, ba, nbx, dt, nm

    tr_rows = np.where(D["split"] == 0)[0]
    for ep in range(epochs):
        model.train(); perm = np.random.default_rng(ep).permutation(len(tr_rows))
        for i in range(0, len(perm), bs):
            rows = tr_rows[perm[i:i + bs]]
            bx, ba, nbx, dt, nm = batch_tensors(rows)
            yv = torch.tensor(D["y_all"][rows], device=device).float()
            logit = model(bx, ba, nbx, dt, nm)
            loss = lossfn(logit, yv)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval(); te_rows = np.where(D["split"] == 1)[0]; scores = []
    with torch.no_grad():
        for i in range(0, len(te_rows), 2048):
            rows = te_rows[i:i + 2048]
            bx, ba, nbx, dt, nm = batch_tensors(rows)
            scores.append(torch.sigmoid(model(bx, ba, nbx, dt, nm)).cpu().numpy())
    s = np.concatenate(scores); y = D["y_all"][te_rows]
    return metrics(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="artifacts/elliptic_results.json")
    args = ap.parse_args()

    t0 = time.time(); device = torch.device(args.device)
    z = np.load(args.data)
    split, y_all, step = z["split"], z["y"].astype(np.int64), z["step"]
    tr, te = split == 0, split == 1
    xl_all = zscore(z["x_local"], tr); xa_all = zscore(z["x_agg"], tr)
    D = {"xl": xl_all, "xa": xa_all, "xl_all": xl_all, "y_all": y_all, "step_all": step,
         "nbr_idx": z["nbr_idx"], "nbr_mask": z["nbr_mask"], "split": split,
         "y": {"train": y_all[tr], "test": y_all[te]}}

    res = {}
    # ---- LightGBM references ----
    import lightgbm as lgb
    for name, X in [("lgbm_local", xl_all), ("lgbm_localagg", np.hstack([xl_all, xa_all]))]:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                               class_weight="balanced", verbose=-1).fit(X[tr], y_all[tr])
        res[name] = metrics(y_all[te], m.predict_proba(X[te])[:, 1])
        print(f"[elliptic] {name:16s} PR {res[name]['pr_auc']:.3f} ROC {res[name]['roc_auc']:.3f} "
              f"illicit-F1 {res[name]['illicit_f1']:.3f}")

    # ---- torch arms ----
    for arm in ["local", "local+agg", "local+meanpool", "local+xseq", "local+agg+xseq"]:
        res[arm] = run_torch(arm, D, device, args.epochs, args.batch_size, args.lr)
        print(f"[elliptic] {arm:16s} PR {res[arm]['pr_auc']:.3f} ROC {res[arm]['roc_auc']:.3f} "
              f"illicit-F1 {res[arm]['illicit_f1']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": "elliptic", "test_base": float(y_all[te].mean()),
                                          "arms": res}, indent=2))
    print(f"[elliptic] wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
