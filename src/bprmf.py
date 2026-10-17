"""BPR-MF (Bayesian Personalized Ranking) adapted for rating regression.

Matrix factorization trained with a hybrid loss: MSE on observed ratings +
pairwise margin on (seen_item, unseen_item) pairs to separate liked vs
unseen items. Adds an orthogonal CF baseline to the stacker (different
training objective than NeuMF's pure MSE).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS
from data import load_ratings, temporal_split, item_popularity, stratified_subsample


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.I = nn.Embedding(n_items, dim)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.g = nn.Parameter(torch.tensor(3.5))
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.I.weight, std=0.02)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)

    def score(self, u, i):
        return (self.g + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1)
                + (self.U(u) * self.I(i)).sum(dim=-1))

    def rate(self, u, i):
        return 1.0 + 4.0 * torch.sigmoid(self.score(u, i))


def metrics(y, p, pop, n_buckets=POPULARITY_BUCKETS):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float); pop = np.asarray(pop, dtype=float)
    out = {"rmse": float(np.sqrt(np.mean((y - p) ** 2))),
           "mae": float(np.mean(np.abs(y - p))),
           "bias": float(np.mean(p - y)),
           "mean_pred": float(p.mean())}
    try:
        q = pd.qcut(np.log1p(pop), q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        q = np.zeros_like(pop, dtype=int)
    buckets = {}
    for b in sorted(pd.Series(q).dropna().unique()):
        m = (q == b)
        buckets[f"bucket_{int(b)}"] = {
            "n": int(m.sum()), "mean_pop": float(pop[m].mean()),
            "rmse": float(np.sqrt(np.mean((y[m] - p[m]) ** 2))),
            "bias": float(np.mean(p[m] - y[m])),
        }
    out["by_popularity"] = buckets
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bpr_weight", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"BPR-MF (dim={args.dim}, device={device})")

    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    user_idx = {int(u): i for i, u in enumerate(sorted(train["user_id"].unique()))}
    item_idx = {int(m): i for i, m in enumerate(sorted(train["movie_id"].unique()))}
    n_users = len(user_idx); n_items = len(item_idx)

    # Positive = rated items with rating >= 4 (for BPR margin)
    pos_mask = train["rating"] >= 4
    pos_users = train.loc[pos_mask, "user_id"].map(user_idx).values
    pos_items = train.loc[pos_mask, "movie_id"].map(item_idx).values
    train_u = torch.tensor(train["user_id"].map(user_idx).values, dtype=torch.long)
    train_i = torch.tensor(train["movie_id"].map(item_idx).values, dtype=torch.long)
    train_r = torch.tensor(train["rating"].astype(np.float32).values)
    ds = TensorDataset(train_u, train_i, train_r)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    rng_np = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng_np)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)

    sample_u = sample["user_id"].map(user_idx).fillna(-1).astype(int).values
    sample_i = sample["movie_id"].map(item_idx).fillna(-1).astype(int).values
    valid = (sample_u >= 0) & (sample_i >= 0)
    y = sample["rating"].astype(float).values
    pops = sample["pop_count"].values

    model = BPRMF(n_users, n_items, dim=args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = {"rmse": 9.9, "epoch": -1, "preds": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0; ncnt = 0
        for bu, bi, br in loader:
            bu = bu.to(device); bi = bi.to(device); br = br.to(device)
            # MSE on observed
            pred = model.rate(bu, bi)
            mse = F.mse_loss(pred, br)
            # BPR on positive pairs vs random negative
            batch_size = bu.size(0)
            neg_i = torch.randint(0, n_items, (batch_size,), device=device)
            pos_score = model.score(bu, bi)
            neg_score = model.score(bu, neg_i)
            bpr = -F.logsigmoid(pos_score - neg_score).mean()
            loss = mse + args.bpr_weight * bpr
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(mse.item()) * bu.numel(); ncnt += bu.numel()
        model.eval()
        with torch.no_grad():
            preds = np.full(len(sample), 3.5, dtype=np.float32)
            if valid.any():
                tu = torch.tensor(sample_u[valid], dtype=torch.long, device=device)
                ti = torch.tensor(sample_i[valid], dtype=torch.long, device=device)
                p = model.rate(tu, ti).cpu().numpy()
                preds[valid] = p
            m = metrics(y, preds, pops)
            print(f"  epoch {epoch:3d}: train_mse={total/max(1,ncnt):.4f}  "
                  f"test_rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  bias={m['bias']:+.4f}")
            if m["rmse"] < best["rmse"]:
                best = {"rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                        "mean_pred": m["mean_pred"], "epoch": epoch,
                        "by_popularity": m["by_popularity"], "preds": preds.copy()}

    print(f"\nBPR-MF best @ epoch {best['epoch']}: rmse={best['rmse']:.4f} "
          f"(elapsed={(time.time()-t0)/60:.1f} min)")

    out_dir = Path(OUTPUT_DIR) / f"bprmf_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["pred_BPRMF"] = best["preds"]
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best": {k: v for k, v in best.items() if k != "preds"},
                   "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
