"""LLM-as-confidence-router.

For each test pair, the LLM is shown the predictions from multiple base
methods (ItemKNN / NeuMF / CFv2_cal / diverse / soft_dist / LightGCN) and
asked to assign per-method confidence weights that sum to 1. The final
prediction is the weighted combination.

This is a direct attack on the oracle gap: instead of learning a fixed
Ridge blend, the LLM reasons per-pair about which method to trust given
the user, item, and prediction spread.
"""
import argparse
import glob
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


SYSTEM = (
    "You are an expert recommender-systems analyst. You are given a target "
    "user-movie pair and rating predictions from several methods (classical "
    "collaborative filtering, neural CF, and LLM-based methods). Assign a "
    "confidence weight in [0, 1] to each method so the weights sum to 1. "
    "Higher weight means you trust that method more *for this particular pair*. "
    "Consider: is this a popular or niche item? Does the user's history clearly "
    "signal their preference? Do the methods agree or disagree?"
)


def build_prompt(user_row, history_df, target_movie, movies_lookup, stats,
                 preds_dict, pop_count):
    profile = format_user_profile(user_row)
    hist = format_history(history_df.head(10), movies_lookup)
    target = format_movie(target_movie)
    stats_str = format_user_stats(stats)
    preds_str = "\n".join(f"  {name}: {v:.2f}" for name, v in preds_dict.items())
    user_msg = f"""User profile: {profile}
Rating statistics: {stats_str}
Recent rating history (top 10):
{hist}

Target movie: {target}
Target movie popularity (train rating count): {int(pop_count)}

Base-method predicted ratings:
{preds_str}

Output per-method confidence weights. They must be non-negative and sum to 1.

Return JSON:
{{"weights": {{{", ".join(f'"{k}": <float>' for k in preds_dict.keys())}}},
  "reasoning": "one-sentence justification"}}"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _normalize_weights(d, keys):
    out = []
    for k in keys:
        v = d.get(k, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        out.append(max(0.0, v))
    s = sum(out)
    if s <= 1e-6:
        return [1.0 / len(keys)] * len(keys)
    return [v / s for v in out]


def load_latest(pattern, cols):
    paths = sorted(glob.glob(str(OUTPUT_DIR / pattern)))
    if not paths:
        return None
    df = pd.read_csv(paths[-1])
    available = [c for c in cols if c in df.columns]
    if not available:
        return None
    return df[["user_id", "movie_id"] + available]


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
    args = ap.parse_args()

    t0 = time.time()
    movies = load_movies(); users = load_users(); ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)

    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)

    # Load predictions from every available base method
    print("Collecting base-method predictions...")
    bases = [
        ("cf_baseline_" + args.tag + "_*/predictions.csv", ["pred_ItemKNN", "pred_SVD", "pred_UserMean"]),
        ("cfv2_" + args.tag + "_*/predictions.csv", ["pred_cfv2_cal"]),
        ("diverse_" + args.tag + "_*/predictions.csv", ["pred_diverse"]),
        ("neumf_" + args.tag + "_*/predictions.csv", ["pred_NeuMF"]),
        ("lightgcn_" + args.tag + "_*/predictions.csv", ["pred_LightGCN"]),
        ("lightgcn_llminit_" + args.tag + "_*/predictions.csv", ["pred_LightGCN_llminit"]),
        ("bprmf_" + args.tag + "_*/predictions.csv", ["pred_BPRMF"]),
        ("twotower_" + args.tag + "_*/predictions.csv", ["pred_TwoTower"]),
    ]
    merged = sample[["user_id", "movie_id", "rating", "pop_count"]].reset_index(drop=True).copy()
    pred_cols = []
    for pat, cols in bases:
        got = load_latest(pat, cols)
        if got is None:
            continue
        got = got.drop_duplicates(subset=["user_id", "movie_id"])
        merged = merged.merge(got, on=["user_id", "movie_id"], how="left")
        for c in cols:
            if c in merged.columns:
                merged[c] = merged[c].clip(1.0, 5.0)
                pred_cols.append(c)
    print(f"  base methods available: {pred_cols}")
    if len(pred_cols) < 2:
        print("  Need at least 2 base methods; aborting.")
        return

    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}
    users_lookup = {int(uid): users.loc[uid].to_dict() for uid in users.index}
    hist_cache = {int(u): user_history(train, int(u)) for u in sample["user_id"].unique()}
    stats_cache = {int(u): user_stats(train, int(u)) for u in sample["user_id"].unique()}

    rows = merged.to_dict("records")

    def work(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = hist_cache[int(row["user_id"])]
        s = stats_cache[int(row["user_id"])]
        preds_dict = {k.replace("pred_", ""): float(row[k]) for k in pred_cols if pd.notna(row[k])}
        if not preds_dict:
            return {"pred": 3.5, "weights": {}, "err": "no base preds"}
        try:
            obj = chat_json(build_prompt(u, h, m, movies_lookup, s, preds_dict, row["pop_count"]),
                            extra_tag=("confidence_router", tuple(sorted(preds_dict.keys()))))
            w_raw = obj.get("weights", {})
            keys = list(preds_dict.keys())
            weights = _normalize_weights(w_raw, keys)
            pred = sum(weights[j] * preds_dict[k] for j, k in enumerate(keys))
            return {"pred": float(np.clip(pred, 1.0, 5.0)),
                    "weights": {k: weights[j] for j, k in enumerate(keys)}}
        except Exception as e:
            return {"pred": float(np.mean(list(preds_dict.values()))),
                    "weights": {}, "err": str(e)[:140]}

    print(f"Running LLM confidence router on {len(rows)} pairs (base preds: {len(pred_cols)})...")
    with tqdm(total=len(rows), desc="router") as pbar:
        outs = batch_map(work, rows, concurrency=args.concurrency, progress=pbar)

    preds = np.array([o["pred"] for o in outs])
    y = merged["rating"].astype(float).values
    pops = merged["pop_count"].values
    m = metrics(y, preds, pops)
    print(f"\nConfidence router: rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:+.4f} "
          f"mean_pred={m['mean_pred']:.3f} (n={len(rows)}, elapsed={(time.time()-t0)/60:.1f} min)")
    for bk, bv in m["by_popularity"].items():
        print(f"  {bk} (n={bv['n']}, pop={bv['mean_pop']:.1f}): rmse={bv['rmse']:.4f} bias={bv['bias']:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"confidence_router_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = merged.copy()
    outdf["pred_ConfidenceRouter"] = preds
    # Store per-method weights
    all_keys = sorted({k for o in outs for k in o.get("weights", {})})
    for k in all_keys:
        outdf[f"w_{k}"] = [o.get("weights", {}).get(k, np.nan) for o in outs]
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"router": m, "config": vars(args), "base_methods": pred_cols}, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
