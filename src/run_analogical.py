"""Analogical rating via LLM.

Retrieve the K movies most similar to the target from the user's training
history (content similarity). Show the LLM those (movie, rating) pairs as
anchor examples and ask: "Given these analogies, rate the target."

This is retrieval-augmented in-context "learning" in the recsys sense —
the LLM reasons by explicit analogy rather than implicit pattern matching.
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
    format_movie, format_user_stats,
)
from llm import batch_map, chat_json

try:
    from retrieval_utils import retrieve_similar_user_items
    HAVE_RETRIEVAL = True
except Exception:
    HAVE_RETRIEVAL = False


SYSTEM = (
    "You predict movie ratings by explicit analogy. You are shown a handful "
    "of the user's previous ratings for movies that are *content-similar* to "
    "the target (same genre, era, or thematic ground). Reason directly from "
    "these analogies — how did the user rate the closest neighbors? "
    "Interpolate to produce the target rating."
)


def build_prompt(user_row, target_movie, analog_items_df, movies_lookup, stats):
    profile = format_user_profile(user_row)
    target = format_movie(target_movie)
    stats_str = format_user_stats(stats)
    lines = []
    for _, r in analog_items_df.iterrows():
        m = movies_lookup.get(int(r["movie_id"]))
        if m is None:
            continue
        sim_note = ""
        if "sim" in r:
            sim_note = f"  (similarity {r['sim']:.2f})"
        lines.append(f"  - {format_movie(m)} — user rated {int(r['rating'])}/5{sim_note}")
    analogs = "\n".join(lines) if lines else "  (no similar items in history)"
    user_msg = f"""User profile: {profile}
Rating statistics: {stats_str}

Target movie: {target}

The user's previous ratings for movies MOST similar to the target:
{analogs}

Reason by analogy:
1. What do these neighbors tell you about how the user treats this kind of movie?
2. Is the target likely to land at, above, or below the neighbor average?
3. Commit to a rating.

Return JSON:
{{"neighbor_summary": "...",
  "reasoning": "...",
  "rating": <int 1-5>,
  "fine_rating": <float 1-5>}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _parse_rating(obj):
    for k in ("fine_rating", "rating"):
        v = obj.get(k)
        if v is None:
            continue
        try:
            return float(np.clip(float(v), 1.0, 5.0))
        except Exception:
            continue
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
    ap.add_argument("--k", type=int, default=8, help="number of analog items to show")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if not HAVE_RETRIEVAL:
        print("retrieval_utils not available — need embed_movies.py output first.")
        return

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
    hist_cache = {int(u): user_history(train, int(u), max_items=100) for u in sample["user_id"].unique()}
    stats_cache = {int(u): user_stats(train, int(u)) for u in sample["user_id"].unique()}

    rows = sample.reset_index(drop=True).to_dict("records")

    def work(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = hist_cache[int(row["user_id"])]
        s = stats_cache[int(row["user_id"])]
        try:
            analog = retrieve_similar_user_items(int(row["user_id"]), int(row["movie_id"]),
                                                 h, k=args.k)
        except Exception as e:
            return {"pred": 3.5, "err": f"retr: {str(e)[:100]}"}
        try:
            obj = chat_json(build_prompt(u, m, analog, movies_lookup, s),
                            extra_tag=("analogical", args.k))
            return {"pred": _parse_rating(obj), "n_analogs": len(analog)}
        except Exception as e:
            return {"pred": 3.5, "err": str(e)[:140]}

    print(f"Running analogical rating on {len(rows)} pairs (k={args.k})...")
    with tqdm(total=len(rows), desc="analog") as pbar:
        outs = batch_map(work, rows, concurrency=args.concurrency, progress=pbar)

    preds = np.clip(np.array([o["pred"] for o in outs]), 1.0, 5.0)
    y = sample["rating"].astype(float).values
    pops = sample["pop_count"].values
    m_raw = metrics(y, preds, pops)
    print(f"\nAnalogical (raw): rmse={m_raw['rmse']:.4f} mae={m_raw['mae']:.4f} "
          f"bias={m_raw['bias']:+.4f} mean_pred={m_raw['mean_pred']:.3f}")

    preds_cal = preds.copy()
    m_cal = None
    if args.calibrate:
        shift = -m_raw["bias"]
        preds_cal = np.clip(preds + shift, 1.0, 5.0)
        m_cal = metrics(y, preds_cal, pops)
        print(f"Analogical (bias-cal, shift={shift:+.3f}): rmse={m_cal['rmse']:.4f} "
              f"mae={m_cal['mae']:.4f} bias={m_cal['bias']:+.4f} mean_pred={m_cal['mean_pred']:.3f}")

    out_dir = Path(OUTPUT_DIR) / f"analogical_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["pred_Analogical_raw"] = preds
    if args.calibrate:
        outdf["pred_Analogical_cal"] = preds_cal
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"raw": m_raw, "cal": m_cal, "config": vars(args)}, f, indent=2)
    print(f"\nSaved: {out_dir} (elapsed={(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
