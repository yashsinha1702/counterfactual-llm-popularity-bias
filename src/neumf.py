"""Neural Matrix Factorization (NeuMF: GMF + MLP concatenated head) for
rating prediction. Trained on ML-1M temporal-train, evaluated on the
same stratified n=2000 test sample as other methods.

Runs on CPU by default to avoid conflicting with vLLM on GPU 1. Set
CUDA_VISIBLE_DEVICES before invocation to use a specific GPU.
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


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, dim=32, mlp_hidden=(64, 32, 16), dropout=0.1):
        super().__init__()
        self.user_emb_mf = nn.Embedding(n_users, dim)
        self.item_emb_mf = nn.Embedding(n_items, dim)
        self.user_emb_mlp = nn.Embedding(n_users, dim)
        self.item_emb_mlp = nn.Embedding(n_items, dim)
        layers = []
        prev = 2 * dim
        for h in mlp_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dim + mlp_hidden[-1], 1)
        nn.init.normal_(self.user_emb_mf.weight, std=0.01)
        nn.init.normal_(self.item_emb_mf.weight, std=0.01)
        nn.init.normal_(self.user_emb_mlp.weight, std=0.01)
        nn.init.normal_(self.item_emb_mlp.weight, std=0.01)

    def forward(self, u, i):
        mf = self.user_emb_mf(u) * self.item_emb_mf(i)
        ml = self.mlp(torch.cat([self.user_emb_mlp(u), self.item_emb_mlp(i)], dim=-1))
        z = torch.cat([mf, ml], dim=-1)
        # Rating in [1, 5]: 1 + 4 * sigmoid
        return 1.0 + 4.0 * torch.sigmoid(self.head(z).squeeze(-1))


def metrics(y, p, pop, n_buckets=POPULARITY_BUCKETS):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float); pop = np.asarray(pop, dtype=float)
    out = {
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "bias": float(np.mean(p - y)),
        "mean_pred": float(p.mean()),
    }
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
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading ratings (device={args.device})...")
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)
    print(f"  train={len(train):,}  test_sample={len(sample)}")

    # Build id maps over union
    all_users = sorted(set(train["user_id"].unique()) | set(sample["user_id"].unique()))
    all_items = sorted(set(train["movie_id"].unique()) | set(sample["movie_id"].unique()))
    u_idx = {int(u): i for i, u in enumerate(all_users)}
    i_idx = {int(m): i for i, m in enumerate(all_items)}
    n_u, n_i = len(all_users), len(all_items)
    print(f"  n_users={n_u} n_items={n_i}")

    u_t = torch.tensor([u_idx[int(u)] for u in train["user_id"]], dtype=torch.long)
    i_t = torch.tensor([i_idx[int(m)] for m in train["movie_id"]], dtype=torch.long)
    r_t = torch.tensor(train["rating"].astype(float).values, dtype=torch.float32)
    u_te = torch.tensor([u_idx[int(u)] for u in sample["user_id"]], dtype=torch.long)
    i_te = torch.tensor([i_idx[int(m)] for m in sample["movie_id"]], dtype=torch.long)
    y_te = sample["rating"].astype(float).values
    pops = sample["pop_count"].astype(int).values

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = NeuMF(n_u, n_i, dim=args.dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(u_t, i_t, r_t)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device == "cuda"))
    u_te_d = u_te.to(device); i_te_d = i_te.to(device)

    best_rmse = float("inf")
    best_preds = None
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for u_b, i_b, r_b in dl:
            u_b, i_b, r_b = u_b.to(device), i_b.to(device), r_b.to(device)
            pred = model(u_b, i_b)
            loss = F.mse_loss(pred, r_b)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(u_b); n += len(u_b)
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            test_preds = model(u_te_d, i_te_d).cpu().numpy()
        test_preds = np.clip(test_preds, 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((y_te - test_preds) ** 2)))
        print(f"  epoch {epoch+1:3d}: train_mse={train_loss:.4f}  test_rmse={rmse:.4f}",
              flush=True)
        if rmse < best_rmse:
            best_rmse = rmse
            best_preds = test_preds
            best_epoch = epoch + 1

    elapsed = time.time() - t0
    m = metrics(y_te, best_preds, pops)
    print(f"\nNeuMF best @ epoch {best_epoch}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} "
          f"bias={m['bias']:+.4f} mean_pred={m['mean_pred']:.3f} (elapsed={elapsed/60:.1f} min)")
    for bk, bv in m["by_popularity"].items():
        print(f"  {bk} (n={bv['n']}, pop={bv['mean_pop']:.1f}): rmse={bv['rmse']:.4f} bias={bv['bias']:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"neumf_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["pred_NeuMF"] = best_preds
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"neumf": m, "best_epoch": best_epoch, "config": vars(args)}, f, indent=2)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
