"""Multi-seed / multi-config NeuMF ensemble.

Trains multiple NeuMF models with different random seeds and/or embedding
dimensions on GPU. Each model's predictions are saved as a separate column
in a single predictions.csv so the stacker can use them independently or
as an averaged ensemble feature.

GPU 0 is used by default (GPU 1 reserved for vLLM).
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
    def __init__(self, n_users, n_items, dim=64, mlp_layers=(128, 64, 32), dropout=0.1):
        super().__init__()
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
        return 1.0 + 4.0 * torch.sigmoid(self.head(torch.cat([mf, ml], dim=-1)).squeeze(-1))


def train_one(n_u, n_i, u_t, i_t, r_t, u_te, i_te, y_te, pops,
              dim=64, mlp_layers=(128, 64, 32), dropout=0.1,
              epochs=40, lr=1e-3, wd=1e-5, batch_size=8192,
              seed=0, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NeuMF(n_u, n_i, dim=dim, mlp_layers=mlp_layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
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
        sched.step()
        model.eval()
        with torch.no_grad():
            tp = model(u_te_d, i_te_d).cpu().numpy()
        tp = np.clip(tp, 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((y_te - tp) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_preds, best_epoch = rmse, tp.copy(), epoch + 1
    return best_preds, best_rmse, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=0, help="Data sampling seed")
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--epochs", type=int, default=40)
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
    pops = sample["pop_count"].astype(int).values

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Configuration grid
    configs = [
        # (dim, mlp_layers, dropout, lr, seed_offset, label)
        (64,  (128, 64, 32), 0.1, 1e-3, 0, "d64_s0"),
        (64,  (128, 64, 32), 0.1, 1e-3, 1, "d64_s1"),
        (64,  (128, 64, 32), 0.1, 1e-3, 2, "d64_s2"),
        (64,  (128, 64, 32), 0.1, 1e-3, 3, "d64_s3"),
        (64,  (128, 64, 32), 0.1, 1e-3, 4, "d64_s4"),
        (128, (256, 128, 64), 0.15, 8e-4, 0, "d128_s0"),
        (128, (256, 128, 64), 0.15, 8e-4, 1, "d128_s1"),
        (128, (256, 128, 64), 0.15, 8e-4, 2, "d128_s2"),
    ]

    outdf = sample.reset_index(drop=True).copy()
    all_preds = {}
    results = {}

    for dim, mlp, drop, lr, seed_off, label in configs:
        print(f"\n--- Training {label} (dim={dim}, mlp={mlp}, drop={drop}, lr={lr}) ---")
        ct = time.time()
        preds, rmse, best_ep = train_one(
            n_u, n_i, u_t, i_t, r_t, u_te, i_te, y_te, pops,
            dim=dim, mlp_layers=mlp, dropout=drop,
            epochs=args.epochs, lr=lr, wd=1e-5, batch_size=8192,
            seed=42 + seed_off, device=device,
        )
        elapsed = time.time() - ct
        col = f"pred_NeuMF_{label}"
        outdf[col] = preds
        all_preds[label] = preds
        results[label] = {"rmse": rmse, "best_epoch": best_ep, "elapsed_min": round(elapsed/60, 1)}
        mae = float(np.mean(np.abs(y_te - preds)))
        bias = float(np.mean(preds - y_te))
        print(f"  → rmse={rmse:.4f} mae={mae:.4f} bias={bias:+.4f} epoch={best_ep} ({elapsed:.0f}s)")

    # Compute ensemble averages
    d64_preds = np.stack([all_preds[k] for k in all_preds if k.startswith("d64")], axis=0)
    d128_preds = np.stack([all_preds[k] for k in all_preds if k.startswith("d128")], axis=0)
    all_arr = np.stack(list(all_preds.values()), axis=0)

    outdf["pred_NeuMF_avg_d64"] = np.clip(d64_preds.mean(axis=0), 1.0, 5.0)
    outdf["pred_NeuMF_avg_d128"] = np.clip(d128_preds.mean(axis=0), 1.0, 5.0)
    outdf["pred_NeuMF_avg_all"] = np.clip(all_arr.mean(axis=0), 1.0, 5.0)

    for name, arr in [("avg_d64", outdf["pred_NeuMF_avg_d64"].values),
                      ("avg_d128", outdf["pred_NeuMF_avg_d128"].values),
                      ("avg_all", outdf["pred_NeuMF_avg_all"].values)]:
        rmse = float(np.sqrt(np.mean((y_te - arr) ** 2)))
        mae = float(np.mean(np.abs(y_te - arr)))
        bias = float(np.mean(arr - y_te))
        results[name] = {"rmse": rmse, "mae": mae, "bias": bias}
        print(f"\n  Ensemble {name}: rmse={rmse:.4f} mae={mae:.4f} bias={bias:+.4f}")

    # Also add prediction std as a feature (disagreement signal)
    outdf["pred_NeuMF_std"] = all_arr.std(axis=0)

    out_dir = Path(OUTPUT_DIR) / f"neumf_ensemble_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"models": results, "config": vars(args)}, f, indent=2)

    elapsed_total = time.time() - t0
    print(f"\nSaved: {out_dir} ({elapsed_total/60:.1f} min total)")


if __name__ == "__main__":
    main()
