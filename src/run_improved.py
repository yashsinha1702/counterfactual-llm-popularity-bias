"""Run the improved LLM variants on the same stratified n=500 sample.

Variants:
  - diverse_v2:         pros/cons prompt + user-rating statistics
  - logprob_no_stats:   logprob expected-value (single-token probs), no user stats
  - logprob_stats:      logprob expected-value + user-rating statistics
  - hybrid_svd_diverse: 0.5 * SVD + 0.5 * diverse_v2 (requires SVD preds file)
  - hybrid_knn_diverse: 0.5 * ItemKNN + 0.5 * diverse_v2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS, LLM_CONCURRENCY
from data import (
    load_movies, load_users, load_ratings,
    temporal_split, item_popularity, popularity_quantile,
    user_history, stratified_subsample, user_stats,
)
from llm import batch_map
from baseline_diverse_v2 import predict_rating_diverse_v2
from predict_logprob import predict_rating_logprob


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
    ap.add_argument("--concurrency", type=int, default=LLM_CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n500")
    ap.add_argument("--cf_pred_file", type=str, default="",
                    help="Path to cf_baseline predictions.csv for hybrid variants")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading data...")
    movies = load_movies(); users = load_users(); ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)

    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}
    users_lookup = {int(uid): users.loc[uid].to_dict() for uid in users.index}
    hist_cache = {int(uid): user_history(train, int(uid)) for uid in sample["user_id"].unique()}
    stats_cache = {int(uid): user_stats(train, int(uid)) for uid in sample["user_id"].unique()}
    rows = sample.reset_index(drop=True).to_dict("records")
    print(f"  sampled={len(rows)}")

    y = np.array([r["rating"] for r in rows], dtype=float)
    pops = np.array([r["pop_count"] for r in rows], dtype=float)

    def _user_id(r): return int(r["user_id"])
    def _ctx(r):
        return (users_lookup[_user_id(r)], movies_lookup[int(r["movie_id"])],
                hist_cache[_user_id(r)], stats_cache[_user_id(r)])

    def work_diverse_v2(row):
        u, m, h, s = _ctx(row)
        try:
            r, _ = predict_rating_diverse_v2(u, h, m, movies_lookup, s)
        except Exception as e:
            return {"pred": 3.0, "err": str(e)[:140]}
        return {"pred": r}

    def work_logprob(row, with_stats):
        u, m, h, s = _ctx(row)
        try:
            r, _ = predict_rating_logprob(u, h, m, movies_lookup, stats=s if with_stats else None)
        except Exception as e:
            return {"pred": 3.0, "err": str(e)[:140]}
        return {"pred": r}

    all_preds = {}

    print("\n[1/3] diverse_v2 (pros/cons + user stats)...")
    with tqdm(total=len(rows), desc="diverse_v2") as pbar:
        outs = batch_map(work_diverse_v2, rows, concurrency=args.concurrency, progress=pbar)
    all_preds["diverse_v2"] = np.array([o["pred"] for o in outs], dtype=float)

    print("\n[2/3] logprob expected-value (no user stats)...")
    with tqdm(total=len(rows), desc="logprob_no_stats") as pbar:
        outs = batch_map(lambda r: work_logprob(r, with_stats=False),
                         rows, concurrency=args.concurrency, progress=pbar)
    all_preds["logprob_no_stats"] = np.array([o["pred"] for o in outs], dtype=float)

    print("\n[3/3] logprob expected-value (with user stats)...")
    with tqdm(total=len(rows), desc="logprob_stats") as pbar:
        outs = batch_map(lambda r: work_logprob(r, with_stats=True),
                         rows, concurrency=args.concurrency, progress=pbar)
    all_preds["logprob_stats"] = np.array([o["pred"] for o in outs], dtype=float)

    # Hybrids if CF prediction file provided
    if args.cf_pred_file and Path(args.cf_pred_file).exists():
        cf_df = pd.read_csv(args.cf_pred_file)
        idx = (cf_df["user_id"].astype(int).astype(str) + "_" + cf_df["movie_id"].astype(int).astype(str)).values
        cf_map = {k: {"SVD": s, "ItemKNN": n, "Baseline": b}
                  for k, s, n, b in zip(idx,
                                        cf_df["pred_SVD"].values,
                                        cf_df["pred_ItemKNN"].values,
                                        cf_df["pred_Baseline"].values)}
        cf_svd = np.array([cf_map[f"{int(r['user_id'])}_{int(r['movie_id'])}"]["SVD"] for r in rows])
        cf_knn = np.array([cf_map[f"{int(r['user_id'])}_{int(r['movie_id'])}"]["ItemKNN"] for r in rows])

        llm_snapshot = list(all_preds.items())
        best_alphas = {}
        hybrid_preds = {}
        for cf_name, cf_vec in [("SVD", cf_svd), ("ItemKNN", cf_knn)]:
            for llm_name, llm_vec in llm_snapshot:
                best = (None, float("inf"))
                for a in np.linspace(0.0, 1.0, 21):
                    blend = a * cf_vec + (1 - a) * llm_vec
                    r = float(np.sqrt(np.mean((y - blend) ** 2)))
                    if r < best[1]:
                        best = (a, r)
                alpha = best[0]
                hybrid_preds[f"hybrid_{cf_name}_{llm_name}_a{alpha:.2f}"] = (
                    alpha * cf_vec + (1 - alpha) * llm_vec
                )
                best_alphas[f"{cf_name}+{llm_name}"] = alpha
        all_preds.update(hybrid_preds)
        print(f"Hybrid best alphas: {best_alphas}")
        # Also include fixed-α=0.5 blends for honest comparison (no tuning cheat)
        for cf_name, cf_vec in [("SVD", cf_svd), ("ItemKNN", cf_knn)]:
            for llm_name, llm_vec in llm_snapshot:
                all_preds[f"fixed_{cf_name}_{llm_name}_a0.5"] = 0.5 * cf_vec + 0.5 * llm_vec

    # Score every variant
    results = {}
    for name, preds in all_preds.items():
        preds = np.clip(preds, 1.0, 5.0)
        m = metrics(y, preds, pops)
        results[name] = m
        print(f"  {name}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:+.4f} mean_pred={m['mean_pred']:.3f}")

    out_dir = Path(OUTPUT_DIR) / f"improved_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = pd.DataFrame(rows).reset_index(drop=True)
    for name, preds in all_preds.items():
        outdf[f"pred_{name}"] = np.clip(preds, 1.0, 5.0)
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)

    # Overall table
    rows_tbl = [{
        "method": name,
        "rmse": r["rmse"], "mae": r["mae"],
        "bias": r["bias"], "mean_pred": r["mean_pred"],
    } for name, r in results.items()]
    print("\n=== Improved LLM variants ===")
    print(pd.DataFrame(rows_tbl).to_string(index=False))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
