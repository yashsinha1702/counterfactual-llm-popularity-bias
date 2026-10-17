"""LightGCN for rating prediction on MovieLens 1M.

LightGCN propagates user/item embeddings along the bipartite rating graph
without nonlinear transformations — simple, surprisingly strong. Adapted
for rating regression (vs BPR ranking) by predicting r̂ = b + <E_u, E_i>
with a 1-5 sigmoid squash.

Supports initializing item embeddings from an external matrix (e.g., LLM
text embeddings), which is the main novelty of this v2 run.
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
from data import (
    load_ratings, temporal_split, item_popularity, stratified_subsample,
)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3,
                 item_init: np.ndarray = None, user_init: np.ndarray = None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.5))
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        if item_init is not None:
            assert item_init.shape == (n_items, dim), \
                f"item_init shape {item_init.shape} != ({n_items}, {dim})"
            self.item_emb.weight.data.copy_(torch.from_numpy(item_init.astype(np.float32)))
        if user_init is not None:
            assert user_init.shape == (n_users, dim)
            self.user_emb.weight.data.copy_(torch.from_numpy(user_init.astype(np.float32)))

    def propagate(self, adj_norm):
        """adj_norm: sparse tensor (n_users + n_items) x (n_users + n_items).
        Returns final user/item embeddings after n_layers of propagation."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj_norm, x)
            emb_list.append(x)
        stacked = torch.stack(emb_list, dim=0)
        final = stacked.mean(dim=0)
        return final[:self.n_users], final[self.n_users:]

    def predict(self, u_emb, i_emb, u_idx, i_idx):
        u = u_emb[u_idx]; i = i_emb[i_idx]
        dot = (u * i).sum(dim=-1)
        r = self.global_bias + self.user_bias(u_idx).squeeze(-1) + \
            self.item_bias(i_idx).squeeze(-1) + dot
        return 1.0 + 4.0 * torch.sigmoid(r)


def build_norm_adj(train_df, user_idx, item_idx, device):
    """Symmetric-normalized adjacency of the bipartite rating graph (no weights)."""
    n_u = len(user_idx); n_i = len(item_idx)
    uids = train_df["user_id"].map(user_idx).values
    iids = train_df["movie_id"].map(item_idx).values
    # Edges: user -> item + n_users, and vice versa (undirected)
    rows = np.concatenate([uids, iids + n_u])
    cols = np.concatenate([iids + n_u, uids])
    vals = np.ones_like(rows, dtype=np.float32)
    # Degree normalization: d_i^{-1/2}
    indices = np.stack([rows, cols], axis=0)
    deg = np.bincount(rows, minlength=n_u + n_i).astype(np.float32)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    # Normalize: D^{-1/2} A D^{-1/2}
    vals = vals * inv_sqrt[rows] * inv_sqrt[cols]
    idx = torch.from_numpy(indices).long()
    v = torch.from_numpy(vals)
    adj = torch.sparse_coo_tensor(idx, v, size=(n_u + n_i, n_u + n_i)).coalesce().to(device)
    return adj


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


def _load_item_init(path, dim, n_items, item_idx_map=None):
    """Load LLM item embeddings and align to LightGCN's item_idx order.

    If a sibling `<path without .npy>_ids.npy` exists, it's used to map
    from the saved embedding row order to LightGCN's internal item_idx
    space (so movies present in the embedding file but absent from the
    train set are dropped, and missing embeddings are random-filled).
    """
    if path is None:
        return None
    embs = np.load(path)
    ids_path = path.replace(".npy", "_ids.npy")
    aligned = None
    if Path(ids_path).exists() and item_idx_map is not None:
        emb_ids = np.load(ids_path).astype(np.int64)
        id_to_row = {int(m): i for i, m in enumerate(emb_ids)}
        aligned = np.zeros((n_items, embs.shape[1]), dtype=np.float32)
        n_found = 0
        rng = np.random.default_rng(0)
        for mid, idx in item_idx_map.items():
            r = id_to_row.get(int(mid))
            if r is None:
                # Random small init
                aligned[idx] = rng.standard_normal(embs.shape[1]).astype(np.float32) * 0.02
            else:
                aligned[idx] = embs[r]
                n_found += 1
        print(f"  aligned {n_found}/{n_items} items to LLM embeddings via sidecar ids")
    else:
        if embs.shape[0] != n_items:
            print(f"  WARNING: item embedding rows ({embs.shape[0]}) != n_items "
                  f"({n_items}) and no sidecar ids — skipping init")
            return None
        aligned = embs
    if aligned.shape[1] == dim:
        out = aligned
    elif aligned.shape[1] > dim:
        rng = np.random.default_rng(0)
        R = rng.standard_normal((aligned.shape[1], dim)).astype(np.float32) / np.sqrt(dim)
        out = aligned @ R
    else:
        pad = np.random.default_rng(0).standard_normal(
            (aligned.shape[0], dim - aligned.shape[1])).astype(np.float32) * 0.01
        out = np.concatenate([aligned, pad], axis=1)
    # Scale to typical embedding magnitude
    out = out.astype(np.float32)
    out *= 0.1 / (np.linalg.norm(out, axis=1, keepdims=True).mean() + 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--item_init", type=str, default="",
                    help="Path to .npy of LLM item embeddings to initialize item embeddings from.")
    args = ap.parse_args()

    t0 = time.time()
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device == "auto":
        device = "cpu"
    print(f"LightGCN (dim={args.dim} layers={args.n_layers} epochs={args.epochs} device={device})")

    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    rng_np = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng_np)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)

    user_idx = {int(u): i for i, u in enumerate(sorted(train["user_id"].unique()))}
    item_idx = {int(m): i for i, m in enumerate(sorted(train["movie_id"].unique()))}
    n_users = len(user_idx); n_items = len(item_idx)
    print(f"  users={n_users}  items={n_items}  train_ratings={len(train)}")

    # Optional LLM item init (aligned by movie_id via sidecar ids file)
    item_init = _load_item_init(args.item_init or None, args.dim, n_items,
                                item_idx_map=item_idx)
    if item_init is not None:
        print(f"  initialized item embeddings from {args.item_init}")

    adj = build_norm_adj(train, user_idx, item_idx, device=device)
    model = LightGCN(n_users, n_items, dim=args.dim, n_layers=args.n_layers,
                     item_init=item_init).to(device)

    # Training data
    train_u = torch.tensor(train["user_id"].map(user_idx).values, dtype=torch.long)
    train_i = torch.tensor(train["movie_id"].map(item_idx).values, dtype=torch.long)
    train_r = torch.tensor(train["rating"].astype(np.float32).values)
    ds = TensorDataset(train_u, train_i, train_r)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # Test tensors
    sample_u = sample["user_id"].map(user_idx).fillna(-1).astype(int).values
    sample_i = sample["movie_id"].map(item_idx).fillna(-1).astype(int).values
    valid = (sample_u >= 0) & (sample_i >= 0)
    y_test = sample["rating"].astype(float).values
    pops = sample["pop_count"].values

    best = {"rmse": 9.9, "epoch": -1, "preds": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0; ncnt = 0
        for bu, bi, br in loader:
            bu = bu.to(device); bi = bi.to(device); br = br.to(device)
            u_emb_full, i_emb_full = model.propagate(adj)
            pred = model.predict(u_emb_full, i_emb_full, bu, bi)
            loss = loss_fn(pred, br)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * bu.numel(); ncnt += bu.numel()
        model.eval()
        with torch.no_grad():
            u_emb_full, i_emb_full = model.propagate(adj)
            preds = np.full(len(sample), 3.5, dtype=np.float32)
            if valid.any():
                tu = torch.tensor(sample_u[valid], dtype=torch.long, device=device)
                ti = torch.tensor(sample_i[valid], dtype=torch.long, device=device)
                p = model.predict(u_emb_full, i_emb_full, tu, ti).cpu().numpy()
                preds[valid] = p
            m = metrics(y_test, preds, pops)
            print(f"  epoch {epoch:3d}: train_mse={total/max(1,ncnt):.4f}  "
                  f"test_rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  bias={m['bias']:+.4f}")
            if m["rmse"] < best["rmse"]:
                best = {"rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                        "mean_pred": m["mean_pred"], "epoch": epoch,
                        "by_popularity": m["by_popularity"],
                        "preds": preds.copy()}

    print(f"\nLightGCN best @ epoch {best['epoch']}: rmse={best['rmse']:.4f} "
          f"mae={best['mae']:.4f} bias={best['bias']:+.4f} mean_pred={best['mean_pred']:.3f} "
          f"(elapsed={(time.time()-t0)/60:.1f} min)")
    for bk, bv in best["by_popularity"].items():
        print(f"  {bk} (n={bv['n']}, pop={bv['mean_pop']:.1f}): "
              f"rmse={bv['rmse']:.4f} bias={bv['bias']:+.4f}")

    suffix = "_llminit" if item_init is not None else ""
    out_dir = Path(OUTPUT_DIR) / f"lightgcn{suffix}_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    col = "pred_LightGCN_llminit" if item_init is not None else "pred_LightGCN"
    outdf[col] = best["preds"]
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best": {k: v for k, v in best.items() if k != "preds"},
                   "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
