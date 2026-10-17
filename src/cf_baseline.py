"""Classical collaborative-filtering baselines implemented directly on
numpy/scipy (no surprise dependency). Trains on the full temporal-train
split and evaluates on the same stratified n=500 sample used by the LLM
methods, so numbers are directly comparable.

Methods implemented:
  - GlobalMean:        predict μ everywhere
  - UserMean:          predict user's mean rating
  - Baseline:          μ + b_u + b_i (ALS-fit biases)
  - SVD (MF):          rank-k truncated SVD on centered sparse matrix
  - ItemKNN:           cosine-similarity k-nearest-item weighted average
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from config import OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS
from data import (
    load_ratings, temporal_split, item_popularity, stratified_subsample,
)


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


def build_matrix(train, user_idx, item_idx):
    """Return (R sparse matrix, user_means dict, item_means dict, mu)."""
    uid = train["user_id"].map(user_idx).values
    iid = train["movie_id"].map(item_idx).values
    r = train["rating"].astype(float).values
    mu = float(r.mean())
    n_u = len(user_idx); n_i = len(item_idx)
    R = csr_matrix((r, (uid, iid)), shape=(n_u, n_i))
    return R, mu, uid, iid, r


def fit_baselines_als(train, user_idx, item_idx, mu, reg_u=10.0, reg_i=25.0, n_iter=10):
    """ALS for user/item biases: r_ui = mu + b_u + b_i."""
    n_u = len(user_idx); n_i = len(item_idx)
    b_u = np.zeros(n_u); b_i = np.zeros(n_i)
    uid = train["user_id"].map(user_idx).values
    iid = train["movie_id"].map(item_idx).values
    r = train["rating"].astype(float).values
    counts_u = np.bincount(uid, minlength=n_u)
    counts_i = np.bincount(iid, minlength=n_i)
    for _ in range(n_iter):
        # b_i update
        num_i = np.bincount(iid, weights=r - mu - b_u[uid], minlength=n_i)
        b_i = num_i / (reg_i + counts_i)
        # b_u update
        num_u = np.bincount(uid, weights=r - mu - b_i[iid], minlength=n_u)
        b_u = num_u / (reg_u + counts_u)
    return b_u, b_i


def fit_svd(train, user_idx, item_idx, mu, b_u, b_i, k=50):
    """Truncated SVD on the bias-subtracted sparse ratings."""
    n_u = len(user_idx); n_i = len(item_idx)
    uid = train["user_id"].map(user_idx).values
    iid = train["movie_id"].map(item_idx).values
    r = train["rating"].astype(float).values
    residuals = r - mu - b_u[uid] - b_i[iid]
    R = csr_matrix((residuals, (uid, iid)), shape=(n_u, n_i))
    k = min(k, min(R.shape) - 1)
    U, s, Vt = svds(R, k=k)
    # svds returns ascending s; reorder
    order = np.argsort(-s)
    U = U[:, order]; s = s[order]; Vt = Vt[order]
    return U, s, Vt


def predict_svd(user_ids, item_ids, user_idx, item_idx, mu, b_u, b_i, U, s, Vt):
    preds = np.zeros(len(user_ids))
    US = U * s  # broadcast scale cols
    for n, (u, i) in enumerate(zip(user_ids, item_ids)):
        ui = user_idx.get(int(u))
        ii = item_idx.get(int(i))
        res = 0.0
        bu = b_u[ui] if ui is not None else 0.0
        bi = b_i[ii] if ii is not None else 0.0
        if ui is not None and ii is not None:
            res = float(US[ui] @ Vt[:, ii])
        preds[n] = mu + bu + bi + res
    return preds


def fit_itemknn(train, user_idx, item_idx, mu, b_u, b_i, k=40, min_support=5):
    """Precompute item x item cosine similarities on the bias-centered user-item matrix."""
    n_u = len(user_idx); n_i = len(item_idx)
    uid = train["user_id"].map(user_idx).values
    iid = train["movie_id"].map(item_idx).values
    r = train["rating"].astype(float).values
    residuals = r - mu - b_u[uid] - b_i[iid]
    R = csr_matrix((residuals, (uid, iid)), shape=(n_u, n_i)).tocsc()
    # cosine similarity between item columns
    norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).flatten())
    norms[norms == 0] = 1.0
    # Note: dense sim matrix for 3883 items is 3883^2 floats = 120MB — OK
    Rn = R.multiply(1.0 / norms)
    sim = (Rn.T @ Rn).toarray()
    np.fill_diagonal(sim, 0.0)
    return sim, R


def predict_itemknn(user_ids, item_ids, user_idx, item_idx, mu, b_u, b_i, sim, R, k=40):
    R_user = R.tocsr()
    preds = np.zeros(len(user_ids))
    for n, (u, i) in enumerate(zip(user_ids, item_ids)):
        ui = user_idx.get(int(u))
        ii = item_idx.get(int(i))
        bu = b_u[ui] if ui is not None else 0.0
        bi = b_i[ii] if ii is not None else 0.0
        base = mu + bu + bi
        if ui is None or ii is None:
            preds[n] = base
            continue
        # items this user rated (residuals)
        row = R_user.getrow(ui)
        rated_items = row.indices
        if len(rated_items) == 0:
            preds[n] = base
            continue
        sims = sim[ii, rated_items]
        order = np.argsort(-sims)[:k]
        top_sims = sims[order]
        top_items = rated_items[order]
        mask = top_sims > 0
        if not mask.any():
            preds[n] = base
            continue
        top_sims = top_sims[mask]
        top_items = top_items[mask]
        top_residuals = np.array(row[0, top_items].toarray()).flatten()
        denom = np.abs(top_sims).sum()
        if denom == 0:
            preds[n] = base
        else:
            preds[n] = base + float((top_sims * top_residuals).sum() / denom)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n500")
    ap.add_argument("--svd_k", type=int, default=50)
    ap.add_argument("--knn_k", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading ratings...")
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)
    print(f"  train={len(train):,} test_sampled={len(sample)} "
          f"users_train={train['user_id'].nunique()} movies_train={train['movie_id'].nunique()}")

    user_idx = {int(u): i for i, u in enumerate(sorted(train["user_id"].unique()))}
    item_idx = {int(m): i for i, m in enumerate(sorted(train["movie_id"].unique()))}
    mu = float(train["rating"].mean())
    user_mean = train.groupby("user_id")["rating"].mean().to_dict()

    # Fit biases
    t = time.time()
    b_u, b_i = fit_baselines_als(train, user_idx, item_idx, mu)
    print(f"Baselines ALS fit in {time.time()-t:.1f}s")

    # Fit SVD
    t = time.time()
    U, s, Vt = fit_svd(train, user_idx, item_idx, mu, b_u, b_i, k=args.svd_k)
    print(f"SVD (k={args.svd_k}) fit in {time.time()-t:.1f}s")

    # Fit ItemKNN
    t = time.time()
    sim_mat, R_sparse = fit_itemknn(train, user_idx, item_idx, mu, b_u, b_i)
    print(f"ItemKNN cosine sims in {time.time()-t:.1f}s")

    # Predict each method on sample
    y = sample["rating"].astype(float).values
    pops = sample["pop_count"].values
    uids = sample["user_id"].values
    iids = sample["movie_id"].values

    preds_by = {}
    preds_by["GlobalMean"] = np.full(len(sample), mu)
    preds_by["UserMean"] = np.array([user_mean.get(int(u), mu) for u in uids])
    preds_by["Baseline"] = np.array([
        mu + (b_u[user_idx.get(int(u))] if int(u) in user_idx else 0.0)
           + (b_i[item_idx.get(int(i))] if int(i) in item_idx else 0.0)
        for u, i in zip(uids, iids)
    ])
    preds_by["SVD"] = predict_svd(uids, iids, user_idx, item_idx, mu, b_u, b_i, U, s, Vt)
    preds_by["ItemKNN"] = predict_itemknn(uids, iids, user_idx, item_idx, mu, b_u, b_i,
                                          sim_mat, R_sparse, k=args.knn_k)

    # Clip to [1, 5]
    for name in preds_by:
        preds_by[name] = np.clip(preds_by[name], 1.0, 5.0)

    results = {}
    for name, p in preds_by.items():
        m = metrics(y, p, pops)
        results[name] = m
        print(f"  {name:12s}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:+.4f} mean_pred={m['mean_pred']:.3f}")

    out_dir = Path(OUTPUT_DIR) / f"cf_baseline_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    for name, p in preds_by.items():
        outdf[f"pred_{name}"] = p
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "config": {
            "n": args.n, "seed": args.seed, "svd_k": args.svd_k, "knn_k": args.knn_k,
        }}, f, indent=2)

    rows = [{
        "method": name,
        "rmse": r["rmse"], "mae": r["mae"],
        "bias": r["bias"], "mean_pred": r["mean_pred"],
        **{f"{bk}_bias": bv["bias"] for bk, bv in r["by_popularity"].items()},
    } for name, r in results.items()]
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    print("\n=== Bias by popularity bucket ===")
    rows_b = []
    for name, r in results.items():
        rows_b.append({"method": name, **{
            bk: f"{bv['bias']:+.3f}" for bk, bv in r["by_popularity"].items()
        }})
    print(pd.DataFrame(rows_b).to_string(index=False))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
