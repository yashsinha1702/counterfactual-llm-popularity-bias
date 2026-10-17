"""LLM-scored soft rating distribution.

For each test pair, ask the LLM to assign a probability to each rating
{1, 2, 3, 4, 5} (summing to 1.0). Compute E[rating] = Σ k·p(k) and also
emit each probability as a separate feature for the stacker.

Uses the retrieval-augmented context (top-K content-similar history
items) so the LLM has the best information we can give it.
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
    load_movies, load_users, load_ratings, temporal_split, item_popularity,
    stratified_subsample, user_history, user_stats, format_user_profile,
    format_movie, format_history, format_user_stats,
)
from llm import batch_map, chat_json

try:
    from retrieval_utils import retrieve_similar_user_items
    HAVE_RETRIEVAL = True
except Exception:
    HAVE_RETRIEVAL = False


SYSTEM = (
    "You are a careful movie-rating predictor. You output a probability "
    "distribution over {1, 2, 3, 4, 5} for how this user will rate the "
    "target movie, reflecting genuine uncertainty in your estimate. "
    "Probabilities must be non-negative and sum to 1."
)


def build_prompt(user_row, history_df, target_movie, movies_lookup, stats,
                 retrieved_df=None):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_lookup)
    target = format_movie(target_movie)
    stats_str = format_user_stats(stats)
    retr_block = ""
    if retrieved_df is not None and len(retrieved_df):
        retr_block = (
            "\nContent-similar items this user has already rated:\n"
            + format_history(retrieved_df, movies_lookup) + "\n"
        )
    user_msg = f"""User profile: {profile}
Rating statistics: {stats_str}

Recent rating history:
{hist}
{retr_block}
Target movie: {target}

Output a probability distribution over the 5 rating classes. If you are \
confident, probabilities should be concentrated; if not, they can spread out.

Return JSON:
{{"p1": <float>, "p2": <float>, "p3": <float>, "p4": <float>, "p5": <float>,
  "brief": "one-sentence justification"}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _normalize(obj):
    ps = []
    for k in range(1, 6):
        v = obj.get(f"p{k}", 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        ps.append(max(0.0, v))
    s = sum(ps)
    if s <= 1e-6:
        return np.array([0.05, 0.10, 0.40, 0.30, 0.15])
    return np.array(ps) / s


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
    ap.add_argument("--concurrency", type=int, default=LLM_CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--use_retrieval", action="store_true")
    ap.add_argument("--retrieval_k", type=int, default=10)
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
    hist_cache = {int(u): user_history(train, int(u)) for u in sample["user_id"].unique()}
    stats_cache = {int(u): user_stats(train, int(u)) for u in sample["user_id"].unique()}

    retrieval_fn = None
    if args.use_retrieval and HAVE_RETRIEVAL:
        retrieval_fn = retrieve_similar_user_items

    rows = sample.reset_index(drop=True).to_dict("records")

    def work(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = hist_cache[int(row["user_id"])]
        s = stats_cache[int(row["user_id"])]
        retr = None
        if retrieval_fn is not None:
            try:
                retr = retrieval_fn(int(row["user_id"]), int(row["movie_id"]),
                                    h, k=args.retrieval_k)
            except Exception:
                retr = None
        try:
            obj = chat_json(build_prompt(u, h, m, movies_lookup, s, retrieved_df=retr),
                            extra_tag=("soft_distribution",
                                       "retr" if retr is not None else "noretr"))
            ps = _normalize(obj)
            expected = float(sum((k + 1) * p for k, p in enumerate(ps)))
            return {"pred": expected, **{f"p{k+1}": float(ps[k]) for k in range(5)}}
        except Exception as e:
            # Fallback to uniform-ish
            return {"pred": 3.5, "p1": 0.05, "p2": 0.10, "p3": 0.40,
                    "p4": 0.30, "p5": 0.15, "err": str(e)[:140]}

    print(f"Running LLM soft-distribution on {len(rows)} pairs...")
    with tqdm(total=len(rows), desc="soft") as pbar:
        outs = batch_map(work, rows, concurrency=args.concurrency, progress=pbar)

    preds = np.clip(np.array([o["pred"] for o in outs]), 1.0, 5.0)
    y = sample["rating"].astype(float).values
    pops = sample["pop_count"].values
    m = metrics(y, preds, pops)
    print(f"\nLLM soft-dist: rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:+.4f} "
          f"mean_pred={m['mean_pred']:.3f} (n={len(rows)}, elapsed={(time.time()-t0)/60:.1f} min)")
    for bk, bv in m["by_popularity"].items():
        print(f"  {bk} (n={bv['n']}, pop={bv['mean_pop']:.1f}): rmse={bv['rmse']:.4f} bias={bv['bias']:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"soft_dist_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["pred_soft_dist"] = preds
    for k in range(1, 6):
        outdf[f"soft_p{k}"] = [o.get(f"p{k}", np.nan) for o in outs]
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"soft_dist": m, "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
