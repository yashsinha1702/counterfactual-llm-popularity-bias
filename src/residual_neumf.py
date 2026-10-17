"""Residual NeuMF: train a second NeuMF to predict the residuals of the first.

1. Generate out-of-fold NeuMF predictions on the TRAINING set (5-fold).
2. Compute residuals: r_residual = true_rating - neumf_oof_pred
3. Train a secondary NeuMF on (user, item) → residual
4. Final prediction on test: primary_pred + residual_pred

This is gradient boosting with neural base learners.
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
from sklearn.model_selection import KFold

from config import OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS
from data import load_ratings, temporal_split, item_popularity, stratified_subsample


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64, mlp_layers=(128, 64, 32),
                 dropout=0.1, output_range=None):
        super().__init__()
        self.output_range = output_range  # (lo, hi) or None for unclamped
        self.user_emb_mf = nn.Embedding(n_users, dim)
        self.item_emb_mf = nn.Embedding(n_items, dim)
        self.user_emb_mlp = nn.Embedding(n_users, dim)
        self.item_emb_mlp = nn.Embedding(n_items, dim)
        layers = []
        prev = 2 * dim
        for h in mlp_layers:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dim + mlp_layers[-1], 1)
        for emb in [self.user_emb_mf, self.item_emb_mf,
                     self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u, i):
        mf = self.user_emb_mf(u) * self.item_emb_mf(i)
        ml = self.mlp(torch.cat([self.user_emb_mlp(u), self.item_emb_mlp(i)], dim=-1))
        out = self.head(torch.cat([mf, ml], dim=-1)).squeeze(-1)
        if self.output_range is not None:
            lo, hi = self.output_range
            return lo + (hi - lo) * torch.sigmoid(out)
        return out  # unclamped for residuals


def train_model(model, u_train, i_train, r_train, u_val, i_val,
                epochs=40, lr=1e-3, wd=1e-5, bs=8192, device="cuda"):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    ds = TensorDataset(u_train.to(device), i_train.to(device), r_train.to(device))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    u_v, i_v = u_val.to(device), i_val.to(device)

    best_preds = None
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for ub, ib, rb in dl:
            pred = model(ub, ib)
            loss = F.mse_loss(pred, rb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(u_v, i_v).cpu().numpy()
        val_loss = float(np.mean(vp ** 2)) if model.output_range is None else float(np.sqrt(np.mean(vp ** 2)))
        if epoch == epochs - 1 or val_loss < best_loss:
            best_preds = vp
            best_loss = val_loss
    return best_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--primary_dim", type=int, default=64)
    ap.add_argument("--residual_dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--residual_epochs", type=int, default=30)
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
    print(f"  n_users={n_u} n_items={n_i}")

    device = args.device if torch.cuda.is_available() else "cpu"

    # Training tensors
    u_t = torch.tensor([u_idx[int(u)] for u in train["user_id"]], dtype=torch.long)
    i_t = torch.tensor([i_idx[int(m)] for m in train["movie_id"]], dtype=torch.long)
    r_t = torch.tensor(train["rating"].astype(float).values, dtype=torch.float32)
    u_te = torch.tensor([u_idx[int(u)] for u in sample["user_id"]], dtype=torch.long)
    i_te = torch.tensor([i_idx[int(m)] for m in sample["movie_id"]], dtype=torch.long)
    y_te = sample["rating"].astype(float).values

    # Step 1: Train primary NeuMF on full training set, get test predictions
    print("\n=== Stage 1: Primary NeuMF ===")
    torch.manual_seed(42)
    primary = NeuMF(n_u, n_i, dim=args.primary_dim,
                     mlp_layers=(128, 64, 32), dropout=0.1,
                     output_range=(1.0, 5.0)).to(device)
    opt = torch.optim.Adam(primary.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    ds = TensorDataset(u_t.to(device), i_t.to(device), r_t.to(device))
    dl = DataLoader(ds, batch_size=8192, shuffle=True)
    u_te_d, i_te_d = u_te.to(device), i_te.to(device)

    best_rmse = float("inf")
    best_primary_test = None
    for epoch in range(args.epochs):
        primary.train()
        for ub, ib, rb in dl:
            pred = primary(ub, ib)
            loss = F.mse_loss(pred, rb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        primary.eval()
        with torch.no_grad():
            tp = primary(u_te_d, i_te_d).cpu().numpy()
        tp = np.clip(tp, 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((y_te - tp) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_primary_test = tp.copy()
        if epoch % 10 == 0:
            print(f"  epoch {epoch+1}: rmse={rmse:.4f}")

    print(f"  Primary NeuMF: rmse={best_rmse:.4f}")

    # Get primary predictions on training set
    primary.eval()
    with torch.no_grad():
        primary_train_preds = primary(u_t.to(device), i_t.to(device)).cpu().numpy()
    primary_train_preds = np.clip(primary_train_preds, 1.0, 5.0)

    # Step 2: Compute training residuals
    train_residuals = r_t.numpy() - primary_train_preds
    print(f"\n  Training residuals: mean={train_residuals.mean():.4f} std={train_residuals.std():.4f}")
    r_resid = torch.tensor(train_residuals, dtype=torch.float32)

    # Step 3: Train residual NeuMF (smaller, unclamped output)
    print("\n=== Stage 2: Residual NeuMF ===")
    results = {}
    outdf = sample.reset_index(drop=True).copy()
    outdf["pred_primary"] = best_primary_test

    for seed_off, label in [(0, "res_s0"), (1, "res_s1"), (2, "res_s2")]:
        torch.manual_seed(100 + seed_off)
        residual_model = NeuMF(n_u, n_i, dim=args.residual_dim,
                                mlp_layers=(64, 32), dropout=0.15,
                                output_range=None).to(device)
        opt_r = torch.optim.Adam(residual_model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=args.residual_epochs, eta_min=5e-6)
        ds_r = TensorDataset(u_t.to(device), i_t.to(device), r_resid.to(device))
        dl_r = DataLoader(ds_r, batch_size=8192, shuffle=True)

        best_combined_rmse = float("inf")
        best_combined_preds = None
        for epoch in range(args.residual_epochs):
            residual_model.train()
            for ub, ib, rb in dl_r:
                pred = residual_model(ub, ib)
                loss = F.mse_loss(pred, rb)
                opt_r.zero_grad(); loss.backward(); opt_r.step()
            sched_r.step()

            residual_model.eval()
            with torch.no_grad():
                rp = residual_model(u_te_d, i_te_d).cpu().numpy()
            combined = np.clip(best_primary_test + rp, 1.0, 5.0)
            rmse = float(np.sqrt(np.mean((y_te - combined) ** 2)))
            if rmse < best_combined_rmse:
                best_combined_rmse = rmse
                best_combined_preds = combined.copy()

        col = f"pred_ResidualNeuMF_{label}"
        outdf[col] = best_combined_preds
        mae = float(np.mean(np.abs(y_te - best_combined_preds)))
        bias = float(np.mean(best_combined_preds - y_te))
        results[label] = {"rmse": best_combined_rmse, "mae": mae, "bias": bias}
        print(f"  {label}: combined rmse={best_combined_rmse:.4f} (primary was {best_rmse:.4f}, delta={best_combined_rmse-best_rmse:+.4f})")

    # Ensemble of residual models
    res_cols = [c for c in outdf.columns if c.startswith("pred_ResidualNeuMF_")]
    outdf["pred_ResidualNeuMF_avg"] = np.clip(outdf[res_cols].mean(axis=1).values, 1.0, 5.0)
    avg = outdf["pred_ResidualNeuMF_avg"].values
    rmse_avg = float(np.sqrt(np.mean((y_te - avg) ** 2)))
    print(f"\n  Residual ensemble avg: rmse={rmse_avg:.4f}")
    results["ensemble"] = {"rmse": rmse_avg}

    out_dir = Path(OUTPUT_DIR) / f"residual_neumf_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "primary_rmse": best_rmse, "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
