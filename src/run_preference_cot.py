"""Preference chain-of-thought rating prediction.

Structured multi-step reasoning prompt:
  1. LLM identifies the user's top-3 preference dimensions from history
     (e.g., "fast pacing", "serious tone", "strong lead").
  2. Scores the target movie on each of those dimensions (1-5).
  3. Outputs a weighted combination → final rating.

Forces the LLM to factor its prediction through explicit preference
dimensions instead of opaque direct regression.
"""
import argparse
import json
import re
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


SYSTEM = (
    "You are a movie-rating predictor using explicit preference-dimension "
    "reasoning. For each user, you first identify what they care about from "
    "their history (3 concrete dimensions); then you score the target movie "
    "on each dimension; then you compute a weighted rating. Be concrete "
    "and grounded in the history — no generic platitudes."
)


def build_prompt(user_row, history_df, target_movie, movies_lookup, stats):
    profile = format_user_profile(user_row)
    hist = format_history(history_df, movies_lookup)
    target = format_movie(target_movie)
    stats_str = format_user_stats(stats)
    user_msg = f"""User profile: {profile}
Rating statistics: {stats_str}

Rating history:
{hist}

Target movie: {target}

Reason step-by-step:
1. Identify 3 preference dimensions this user cares about (e.g., "fast pacing",
   "ensemble casts", "80s style"). Be specific to THIS user's history.
2. For each dimension, estimate the user's weight on it (0-1, sum ≈ 1).
3. Score the target movie on each dimension (1-5) with brief evidence.
4. Compute the weighted expected rating.

Return JSON:
{{
  "dimensions": [
    {{"name": "...", "user_weight": <float 0-1>, "movie_score": <int 1-5>, "evidence": "..."}},
    ... (3 total)
  ],
  "weighted_rating": <float 1-5>,
  "rating": <integer 1-5>
}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _compute_rating(obj):
    # Prefer the weighted-rating if numerically valid, else compute from dims
    try:
        w = float(obj.get("weighted_rating"))
        if 1.0 <= w <= 5.0:
            return float(np.clip(w, 1.0, 5.0))
    except Exception:
        pass
    dims = obj.get("dimensions", [])
    if dims:
        weights = []; scores = []
        for d in dims:
            try:
                weights.append(float(d.get("user_weight", 0.33)))
                scores.append(float(d.get("movie_score", 3.0)))
            except Exception:
                pass
        s = sum(weights)
        if s > 1e-6 and scores:
            weights = [w / s for w in weights]
            return float(np.clip(sum(w * sc for w, sc in zip(weights, scores)), 1.0, 5.0))
    # Fall back to integer rating field
    try:
        return float(np.clip(float(obj.get("rating", 3.0)), 1.0, 5.0))
    except Exception:
        return 3.5


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
    ap.add_argument("--calibrate", action="store_true")
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

    rows = sample.reset_index(drop=True).to_dict("records")

    def work(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = hist_cache[int(row["user_id"])]
        s = stats_cache[int(row["user_id"])]
        try:
            obj = chat_json(build_prompt(u, h, m, movies_lookup, s),
                            extra_tag=("preference_cot",))
            return {"pred": _compute_rating(obj), "obj": obj}
        except Exception as e:
            return {"pred": 3.5, "err": str(e)[:140]}

    print(f"Running preference CoT on {len(rows)} pairs...")
    with tqdm(total=len(rows), desc="cot") as pbar:
        outs = batch_map(work, rows, concurrency=args.concurrency, progress=pbar)

    preds = np.clip(np.array([o["pred"] for o in outs]), 1.0, 5.0)
    y = sample["rating"].astype(float).values
    pops = sample["pop_count"].values
    m_raw = metrics(y, preds, pops)
    print(f"\nPreference CoT (raw): rmse={m_raw['rmse']:.4f} mae={m_raw['mae']:.4f} "
          f"bias={m_raw['bias']:+.4f} mean_pred={m_raw['mean_pred']:.3f}")

    preds_cal = preds.copy()
    m_cal = None
    if args.calibrate:
        shift = -m_raw["bias"]
        preds_cal = np.clip(preds + shift, 1.0, 5.0)
        m_cal = metrics(y, preds_cal, pops)
        print(f"Preference CoT (bias-cal, shift={shift:+.3f}): rmse={m_cal['rmse']:.4f} "
              f"mae={m_cal['mae']:.4f} bias={m_cal['bias']:+.4f} mean_pred={m_cal['mean_pred']:.3f}")

    out_dir = Path(OUTPUT_DIR) / f"preference_cot_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["pred_PreferenceCoT_raw"] = preds
    if args.calibrate:
        outdf["pred_PreferenceCoT_cal"] = preds_cal
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"raw": m_raw, "cal": m_cal, "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir} (elapsed={(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
