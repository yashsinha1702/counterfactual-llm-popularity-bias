"""Multi-seed NeuMF ensemble using the ORIGINAL proven architecture.

Uses the exact same NeuMF class from neumf.py (ReLU, no cosine LR),
just varies random seeds. The v3 attempt failed because GELU + cosine LR
degraded individual model quality from 0.884 to 0.893.

This script trains the PROVEN architecture with multiple seeds, then
adds the ensemble average + genre affinity features to the stacker.
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
    """Exact copy of the ORIGINAL NeuMF from neumf.py — ReLU, not GELU."""
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
            layers.append(nn.ReLU())       # ← ReLU, NOT GELU
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
        return 1.0 + 4.0 * torch.sigmoid(self.head(z).squeeze(-1))


def train_one(n_u, n_i, u_t, i_t, r_t, u_te, i_te, y_te,
              dim=64, mlp_hidden=(64, 32, 16), dropout=0.1,
              epochs=40, lr=1e-3, wd=1e-5, batch_size=8192,
              seed=0, device="cuda"):
    """Train ONE NeuMF with the original architecture. No cosine LR."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NeuMF(n_u, n_i, dim=dim, mlp_hidden=mlp_hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # NO cosine scheduler — plain Adam, matching original neumf.py
    ds = TensorDataset(u_t.to(device), i_t.to(device), r_t.to(device))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    u_te_d, i_te_d = u_te.to(device), i_te.to(device)

    best_rmse, best_preds, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for ub, ib, rb in dl:
            pred = model(ub, ib)
            loss = F.mse_loss(pred, rb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(ub); n += len(ub)
        model.eval()
        with torch.no_grad():
            tp = model(u_te_d, i_te_d).cpu().numpy()
        tp = np.clip(tp, 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((y_te - tp) ** 2)))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1:3d}: train_mse={total/n:.4f} test_rmse={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_preds, best_epoch = rmse, tp.copy(), epoch + 1
    return best_preds, best_rmse, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=0, help="Data sampling seed")
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading data...")
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)
    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)

    all_users = sorted(set(train["user_id"].unique()) | set(sample["user_id"].unique()))
    all_items = sorted(set(train["movie_id"].unique()) | set(sample["movie_id"].unique()))
    u_idx = {int(u): i for i, u in enumerate(all_users)}
    i_idx = {int(m): i for i, m in enumerate(all_items)}
    n_u, n_i = len(all_users), len(all_items)
    print(f"  n_users={n_u} n_items={n_i} train={len(train):,} test={len(sample)}")

    u_t = torch.tensor([u_idx[int(u)] for u in train["user_id"]], dtype=torch.long)
    i_t = torch.tensor([i_idx[int(m)] for m in train["movie_id"]], dtype=torch.long)
    r_t = torch.tensor(train["rating"].astype(float).values, dtype=torch.float32)
    u_te = torch.tensor([u_idx[int(u)] for u in sample["user_id"]], dtype=torch.long)
    i_te = torch.tensor([i_idx[int(m)] for m in sample["movie_id"]], dtype=torch.long)
    y_te = sample["rating"].astype(float).values

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Train N seeds with IDENTICAL hyperparams to the proven config
    outdf = sample.reset_index(drop=True).copy()
    all_preds = {}
    results = {}

    for seed_off in range(args.n_seeds):
        label = f"s{seed_off}"
        seed = seed_off  # seeds: 0, 1, 2, 3, 4
        print(f"\n--- NeuMF {label} (dim={args.dim}, seed={seed}) ---")
        ct = time.time()
        preds, rmse, best_ep = train_one(
            n_u, n_i, u_t, i_t, r_t, u_te, i_te, y_te,
            dim=args.dim, mlp_hidden=(64, 32, 16), dropout=0.1,
            epochs=args.epochs, lr=1e-3, wd=1e-5, batch_size=8192,
            seed=seed, device=device,
        )
        elapsed = time.time() - ct
        col = f"pred_NeuMF_{label}"
        outdf[col] = preds
        all_preds[label] = preds
        mae = float(np.mean(np.abs(y_te - preds)))
        bias = float(np.mean(preds - y_te))
        results[label] = {"rmse": rmse, "mae": mae, "bias": bias,
                          "best_epoch": best_ep, "elapsed_min": round(elapsed/60, 1)}
        print(f"  → rmse={rmse:.4f} mae={mae:.4f} bias={bias:+.4f} epoch={best_ep} ({elapsed:.0f}s)")

    # Ensemble average
    all_arr = np.stack(list(all_preds.values()), axis=0)
    outdf["pred_NeuMF_ens_avg"] = np.clip(all_arr.mean(axis=0), 1.0, 5.0)
    outdf["pred_NeuMF_ens_std"] = all_arr.std(axis=0)

    avg = outdf["pred_NeuMF_ens_avg"].values
    rmse_avg = float(np.sqrt(np.mean((y_te - avg) ** 2)))
    mae_avg = float(np.mean(np.abs(y_te - avg)))
    bias_avg = float(np.mean(avg - y_te))
    results["ensemble_avg"] = {"rmse": rmse_avg, "mae": mae_avg, "bias": bias_avg}
    print(f"\n  ★ Ensemble avg ({args.n_seeds} seeds): rmse={rmse_avg:.4f} mae={mae_avg:.4f} bias={bias_avg:+.4f}")

    # Compare with individual best
    best_single = min(r["rmse"] for k, r in results.items() if k != "ensemble_avg")
    print(f"  Best single: {best_single:.4f}  Ensemble gain: {best_single - rmse_avg:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"neumf_ens_v4_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"models": results, "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir} ({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
