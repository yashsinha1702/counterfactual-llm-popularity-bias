"""Evaluate the diverse-baseline on the same sampled pairs as run.py."""
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
    user_history, stratified_subsample,
)
from llm import batch_map
from baseline_diverse import predict_rating_diverse_baseline


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
    args = ap.parse_args()

    t0 = time.time()
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
    rows = sample.reset_index(drop=True).to_dict("records")

    def work(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = hist_cache[int(row["user_id"])]
        try:
            r, reason = predict_rating_diverse_baseline(u, h, m, movies_lookup)
        except Exception as e:
            return {"pred": 3.0, "err": str(e)[:160]}
        return {"pred": r, "reason": reason}

    print(f"Running diverse_baseline on {len(rows)} pairs...")
    with tqdm(total=len(rows), desc="diverse") as pbar:
        outs = batch_map(work, rows, concurrency=args.concurrency, progress=pbar)

    y = np.array([r["rating"] for r in rows])
    pops = np.array([r["pop_count"] for r in rows])
    preds = np.array([o["pred"] for o in outs])
    m = metrics(y, preds, pops)

    elapsed = time.time() - t0
    print(f"\ndiverse_baseline: rmse={m['rmse']:.4f} mae={m['mae']:.4f} "
          f"bias={m['bias']:+.4f} mean_pred={m['mean_pred']:.3f} (n={len(rows)}, elapsed={elapsed/60:.1f} min)")
    print("\nBias by popularity bucket:")
    for bk, bv in m["by_popularity"].items():
        print(f"  {bk} (n={bv['n']}, mean_pop={bv['mean_pop']:.1f}): "
              f"rmse={bv['rmse']:.4f} bias={bv['bias']:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"diverse_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(pred_diverse=preds).to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"diverse_baseline": m, "config": {"n": args.n, "seed": args.seed}}, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
