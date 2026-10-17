"""Two-tower rating predictor: user tower and item tower trained jointly.

User tower input: fused features derived from user's rating history
(mean item embedding weighted by rating, plus demographic embedding).
Item tower input: LLM item embedding (from embed_llm_items.py).

Both towers project into a shared d-dim space; rating = 1 + 4*sigmoid(
  global_bias + user_bias + item_bias + <U, I>
).

The key novelty vs NeuMF: item representation comes from an LLM-derived
content embedding, not from random init. The user representation is built
from *retrieved* item embeddings weighted by the user's past ratings —
a purely-text view of the user's taste.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS
from data import load_ratings, temporal_split, item_popularity, stratified_subsample


class TwoTower(nn.Module):
    def __init__(self, item_emb_matrix, user_dim_in, hidden=64, out_dim=32,
                 n_users=6041, n_items=3884, dropout=0.2):
        super().__init__()
        n, d = item_emb_matrix.shape
        self.item_base = nn.Embedding.from_pretrained(
            torch.from_numpy(item_emb_matrix.astype(np.float32)), freeze=True,
        )
        self.item_proj = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.user_proj = nn.Sequential(
            nn.Linear(user_dim_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.global_bias = nn.Parameter(torch.tensor(3.5))

    def forward(self, user_feat, item_idx, user_idx):
        u = self.user_proj(user_feat)
        i = self.item_proj(self.item_base(item_idx))
        dot = (u * i).sum(dim=-1)
        r = (self.global_bias + self.user_bias(user_idx).squeeze(-1)
             + self.item_bias(item_idx).squeeze(-1) + dot)
        return 1.0 + 4.0 * torch.sigmoid(r)


def build_user_features(train, item_emb, item_idx):
    """For each user, compute a rating-weighted mean of their rated items'
    LLM embeddings, concatenated with count & mean-rating scalars."""
    d = item_emb.shape[1]
    users = sorted(train["user_id"].unique())
    out = np.zeros((len(users), d + 2), dtype=np.float32)
    uid_map = {int(u): i for i, u in enumerate(users)}
    # Weighted sum per user
    train_by_user = train.groupby("user_id")
    for uid, g in train_by_user:
        r = uid_map[int(uid)]
        item_inds = g["movie_id"].map(item_idx).dropna().astype(int).values
        ratings = g["rating"].astype(np.float32).values[:len(item_inds)]
        if len(item_inds) == 0:
            continue
        # Subtract 3 from ratings so signed weight
        w = (ratings - 3.0)
        # Avoid all-zero normalization
        wabs = np.abs(w).sum()
        if wabs < 1e-6:
            vec = item_emb[item_inds].mean(axis=0)
        else:
            vec = (item_emb[item_inds] * w[:, None]).sum(axis=0) / max(1, len(item_inds))
        out[r, :d] = vec
        out[r, d] = float(np.log1p(len(item_inds)))
        out[r, d + 1] = float(ratings.mean())
    return users, uid_map, out


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
    ap.add_argument("--item_emb", type=str,
                    default=str(OUTPUT_DIR / "item_llm_emb.npy"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out_dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    t0 = time.time()
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device == "auto":
        device = "cpu"
    print(f"TwoTower (item_emb={args.item_emb} device={device})")

    item_emb = np.load(args.item_emb)
    print(f"  item embedding shape: {item_emb.shape}")

    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    movie_ids = np.load(args.item_emb.replace(".npy", "_ids.npy"))
    item_idx_map = {int(m): i for i, m in enumerate(movie_ids)}

    # User features from rating-weighted item embeddings
    users_list, user_idx_map, user_features = build_user_features(train, item_emb, item_idx_map)
    print(f"  users={len(users_list)}  user_feat_dim={user_features.shape[1]}")

    # Stratified sample
    rng_np = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng_np)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)

    # Map all rows to indices
    n_users = max(user_idx_map.values()) + 1
    n_items = max(item_idx_map.values()) + 1
    train_u = train["user_id"].map(user_idx_map).values
    train_i = train["movie_id"].map(item_idx_map).values
    keep = (~pd.isna(train_u)) & (~pd.isna(train_i))
    train = train[keep]
    train_u = train_u[keep].astype(int)
    train_i = train_i[keep].astype(int)
    train_r = train["rating"].astype(np.float32).values

    user_feat_tensor = torch.from_numpy(user_features)

    # DataLoader
    ds = TensorDataset(
        torch.tensor(train_u, dtype=torch.long),
        torch.tensor(train_i, dtype=torch.long),
        torch.tensor(train_r, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TwoTower(item_emb_matrix=item_emb, user_dim_in=user_features.shape[1],
                     hidden=args.hidden, out_dim=args.out_dim,
                     n_users=n_users, n_items=n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Test prep
    sample_u = sample["user_id"].map(user_idx_map).values
    sample_i = sample["movie_id"].map(item_idx_map).values
    valid = (~pd.isna(sample_u)) & (~pd.isna(sample_i))
    sample_u = pd.Series(sample_u).fillna(0).astype(int).values
    sample_i = pd.Series(sample_i).fillna(0).astype(int).values
    y = sample["rating"].astype(float).values
    pops = sample["pop_count"].values

    best = {"rmse": 9.9, "epoch": -1, "preds": None}
    user_feat_dev = user_feat_tensor.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0; ncnt = 0
        for bu, bi, br in loader:
            bu = bu.to(device); bi = bi.to(device); br = br.to(device)
            uf = user_feat_dev[bu]
            pred = model(uf, bi, bu)
            loss = loss_fn(pred, br)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * bu.numel(); ncnt += bu.numel()
        model.eval()
        with torch.no_grad():
            preds = np.full(len(sample), 3.5, dtype=np.float32)
            if valid.any():
                mask = valid.values if hasattr(valid, "values") else valid
                u_t = torch.tensor(sample_u[mask], dtype=torch.long, device=device)
                i_t = torch.tensor(sample_i[mask], dtype=torch.long, device=device)
                uf = user_feat_dev[u_t]
                p = model(uf, i_t, u_t).cpu().numpy()
                preds[mask] = p
            m = metrics(y, preds, pops)
            print(f"  epoch {epoch:3d}: train_mse={total/max(1,ncnt):.4f}  "
                  f"test_rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  bias={m['bias']:+.4f}")
            if m["rmse"] < best["rmse"]:
                best = {"rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                        "mean_pred": m["mean_pred"], "epoch": epoch,
                        "by_popularity": m["by_popularity"], "preds": preds.copy()}

    print(f"\nTwoTower best @ epoch {best['epoch']}: rmse={best['rmse']:.4f} "
          f"mae={best['mae']:.4f} bias={best['bias']:+.4f} (elapsed={(time.time()-t0)/60:.1f} min)")

    out_dir = Path(OUTPUT_DIR) / f"twotower_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["pred_TwoTower"] = best["preds"]
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best": {k: v for k, v in best.items() if k != "preds"},
                   "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
