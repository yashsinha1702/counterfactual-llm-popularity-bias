"""Cold-start experiment: sample 500 test pairs restricted to items with
very low training popularity (default pop<=20), then run the winning CF
method (ItemKNN), SVD, UserMean, and the winning LLM method (diverse_v2)
on them. Answers: does the LLM beat CF on truly-cold items?"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm

from config import OUTPUT_DIR, LLM_CONCURRENCY, POPULARITY_BUCKETS
from data import (
    load_movies, load_users, load_ratings,
    temporal_split, item_popularity, user_history, user_stats,
)
from llm import batch_map
from baseline_diverse_v2 import predict_rating_diverse_v2
from cf_baseline import (
    fit_baselines_als, fit_svd, fit_itemknn,
    predict_svd, predict_itemknn,
)


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def bias(y, p): return float(np.mean(p - y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pop_max", type=int, default=20, help="items with pop<=this in train")
    ap.add_argument("--concurrency", type=int, default=LLM_CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="coldstart")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading...")
    movies = load_movies()
    users = load_users()
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    # Cold items: pop <= pop_max in train. Also include items that never
    # appear in train at all (true cold-start).
    test_with_pop = test.assign(pop_count=test["movie_id"].map(pop).fillna(0).astype(int))
    cold = test_with_pop[test_with_pop["pop_count"] <= args.pop_max]
    print(f"  test pairs with item train_pop <= {args.pop_max}: {len(cold)} "
          f"({(cold['pop_count'] == 0).sum()} unseen items)")
    if len(cold) > args.n:
        cold = cold.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"  sampled: {len(cold)}  mean_pop={cold['pop_count'].mean():.1f}")

    # --- Classical CF fits on FULL train ---
    print("Fitting CF models on full train...")
    user_idx = {int(u): i for i, u in enumerate(sorted(train["user_id"].unique()))}
    item_idx = {int(m): i for i, m in enumerate(sorted(train["movie_id"].unique()))}
    mu = float(train["rating"].mean())
    user_mean_map = train.groupby("user_id")["rating"].mean().to_dict()
    b_u, b_i = fit_baselines_als(train, user_idx, item_idx, mu)
    U, s, Vt = fit_svd(train, user_idx, item_idx, mu, b_u, b_i, k=50)
    sim_mat, R_sparse = fit_itemknn(train, user_idx, item_idx, mu, b_u, b_i)

    uids = cold["user_id"].values
    iids = cold["movie_id"].values
    y = cold["rating"].astype(float).values
    pops = cold["pop_count"].values

    preds = {}
    preds["GlobalMean"] = np.full(len(cold), mu)
    preds["UserMean"] = np.array([user_mean_map.get(int(u), mu) for u in uids])
    preds["Baseline_CF"] = np.array([
        mu + (b_u[user_idx.get(int(u))] if int(u) in user_idx else 0.0)
           + (b_i[item_idx.get(int(i))] if int(i) in item_idx else 0.0)
        for u, i in zip(uids, iids)
    ])
    preds["SVD"] = predict_svd(uids, iids, user_idx, item_idx, mu, b_u, b_i, U, s, Vt)
    preds["ItemKNN"] = predict_itemknn(uids, iids, user_idx, item_idx, mu, b_u, b_i,
                                       sim_mat, R_sparse, k=40)
    for name in list(preds.keys()):
        preds[name] = np.clip(preds[name], 1.0, 5.0)

    # --- LLM: diverse_v2 ---
    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}
    users_lookup = {int(uid): users.loc[uid].to_dict() for uid in users.index}
    print("Caching user history + stats...")
    hist_cache = {int(uid): user_history(train, int(uid)) for uid in cold["user_id"].unique()}
    stats_cache = {int(uid): user_stats(train, int(uid)) for uid in cold["user_id"].unique()}
    rows = cold.reset_index(drop=True).to_dict("records")

    def work(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = hist_cache[int(row["user_id"])]
        s = stats_cache[int(row["user_id"])]
        try:
            r, _ = predict_rating_diverse_v2(u, h, m, movies_lookup, s)
        except Exception as e:
            return {"pred": 3.0, "err": str(e)[:140]}
        return {"pred": r}

    print(f"Running diverse_v2 LLM on {len(rows)} pairs...")
    with tqdm(total=len(rows), desc="diverse_v2") as pbar:
        outs = batch_map(work, rows, concurrency=args.concurrency, progress=pbar)
    preds["diverse_v2_14B"] = np.clip(
        np.array([o["pred"] for o in outs], dtype=float), 1.0, 5.0
    )

    # --- Hybrid ---
    preds["hybrid_KNN_diverse_05"] = np.clip(
        0.5 * preds["ItemKNN"] + 0.5 * preds["diverse_v2_14B"], 1.0, 5.0
    )

    # --- Score ---
    print("\n=== Cold-start results (train pop ≤ %d, n=%d) ===" % (args.pop_max, len(cold)))
    rows_tbl = []
    for name, p in preds.items():
        rows_tbl.append({
            "method": name, "rmse": rmse(y, p), "mae": mae(y, p), "bias": bias(y, p),
            "mean_pred": float(p.mean()),
        })
    table = pd.DataFrame(rows_tbl).sort_values("rmse").reset_index(drop=True)
    print(table.to_string(index=False))

    out_dir = Path(OUTPUT_DIR) / f"coldstart_pop{args.pop_max}_n{len(cold)}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = cold.reset_index(drop=True).copy()
    for name, p in preds.items():
        outdf[f"pred_{name}"] = p
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    table.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "config": vars(args), "n_sampled": int(len(cold)),
            "mean_pop": float(cold["pop_count"].mean()),
            "results": rows_tbl,
        }, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
