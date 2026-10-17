import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS,
    TIPS_WINDOW_DAYS, K_EXPLANATIONS, LLM_CONCURRENCY,
)
from data import (
    load_movies, load_users, load_ratings,
    temporal_split, item_popularity, popularity_quantile, build_tips_table,
    user_history, stratified_subsample,
)
from llm import batch_map
from abduction import generate_candidates
from intervention import score_hypotheses, select_debiased
from prediction import predict_rating
from baseline import predict_rating_baseline


def predict_cf(user_row, history_df, target_movie, movies_df,
               prop_table, window_secs, bucket_max, pop_quant,
               target_mid, target_ts, k):
    hyps = generate_candidates(user_row, history_df, target_movie, movies_df, k=k)
    scored = score_hypotheses(user_row, history_df, target_movie, movies_df, hyps)
    best, all_scored = select_debiased(
        scored, target_mid, target_ts,
        prop_table, window_secs, bucket_max, pop_quant,
    )
    r, brief = predict_rating(user_row, target_movie, best["text"])
    return {
        "rating": r,
        "brief": brief,
        "chosen_id": best["id"],
        "chosen_text": best["text"],
        "chosen_direction": best.get("direction"),
        "chosen_plausibility": best.get("plausibility"),
        "chosen_popularity_reliance": best.get("popularity_reliance"),
        "tips_pop_score": best.get("tips_pop_score"),
        "all_scored": all_scored,
    }


def metrics(df, pred_col, n_buckets=POPULARITY_BUCKETS):
    y = df["rating"].astype(float).values
    p = df[pred_col].astype(float).values
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    bias = float(np.mean(p - y))
    out = {"rmse": rmse, "mae": mae, "mean_pred": float(p.mean()),
           "mean_true": float(y.mean()), "bias": bias, "n": int(len(df))}
    pop = df["pop_count"].values
    try:
        q = pd.qcut(np.log1p(pop), q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        q = np.zeros_like(pop, dtype=int)
    strat = {}
    for b in sorted(pd.Series(q).dropna().unique()):
        m = q == b
        if not m.any():
            continue
        strat[f"bucket_{int(b)}"] = {
            "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean((y[m] - p[m]) ** 2))),
            "mae": float(np.mean(np.abs(y[m] - p[m]))),
            "bias": float(np.mean(p[m] - y[m])),
            "mean_pop": float(pop[m].mean()),
        }
    out["by_popularity"] = strat
    return out


def run(n=SUBSAMPLE_N, k=K_EXPLANATIONS, concurrency=LLM_CONCURRENCY,
        seed=None, tag="shake"):
    t0 = time.time()
    print("Loading data...")
    movies = load_movies()
    users = load_users()
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)
    pop_quant = popularity_quantile(pop).to_dict()
    prop_table, window_secs, bucket_max = build_tips_table(train, window_days=TIPS_WINDOW_DAYS)
    print(f"  train={len(train):,} test={len(test):,} movies={len(movies):,} users={len(users):,}")

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    sample = stratified_subsample(test, pop, n=n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)
    print(f"  subsample={len(sample)}")

    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}
    users_lookup = {int(uid): users.loc[uid].to_dict() for uid in users.index}
    hist_cache = {}
    for uid in sample["user_id"].unique():
        hist_cache[int(uid)] = user_history(train, int(uid))
    def get_hist(uid):
        return hist_cache.get(int(uid), user_history(train, int(uid)))

    rows = sample.reset_index(drop=True).to_dict("records")

    def work_baseline(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = get_hist(row["user_id"])
        try:
            r, reason = predict_rating_baseline(u, h, m, movies_lookup)
        except Exception as e:
            return {"pred_baseline": 3.0, "baseline_err": str(e)[:200]}
        return {"pred_baseline": r, "baseline_reason": reason}

    def work_cf(row):
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = get_hist(row["user_id"])
        try:
            o = predict_cf(u, h, m, movies_lookup, prop_table, window_secs,
                           bucket_max, pop_quant,
                           row["movie_id"], row["timestamp"], k=k)
        except Exception as e:
            return {"pred_cf": 3.0, "cf_err": str(e)[:200]}
        return {
            "pred_cf": o["rating"],
            "cf_brief": o["brief"],
            "cf_chosen_id": o["chosen_id"],
            "cf_chosen_text": o["chosen_text"],
            "cf_chosen_plausibility": o["chosen_plausibility"],
            "cf_chosen_popreliance": o["chosen_popularity_reliance"],
            "cf_tips_pop_score": o["tips_pop_score"],
        }

    print(f"Running baseline (n={len(rows)}, conc={concurrency})...")
    with tqdm(total=len(rows), desc="baseline") as pbar:
        base_out = batch_map(work_baseline, rows, concurrency=concurrency, progress=pbar)

    print(f"Running counterfactual pipeline (n={len(rows)}, conc={concurrency})...")
    with tqdm(total=len(rows), desc="cf") as pbar:
        cf_out = batch_map(work_cf, rows, concurrency=concurrency, progress=pbar)

    df = pd.DataFrame(rows)
    base_keys = set().union(*[o.keys() for o in base_out])
    cf_keys = set().union(*[o.keys() for o in cf_out])
    for col in base_keys:
        df[col] = [b.get(col) for b in base_out]
    for col in cf_keys:
        df[col] = [c.get(col) for c in cf_out]
    df["pred_baseline"] = df["pred_baseline"].fillna(3.0)
    df["pred_cf"] = df["pred_cf"].fillna(3.0)

    m_base = metrics(df, "pred_baseline")
    m_cf = metrics(df, "pred_cf")
    elapsed = time.time() - t0

    print("\n=== RESULTS ===")
    print(f"baseline:      rmse={m_base['rmse']:.4f} mae={m_base['mae']:.4f} "
          f"bias={m_base['bias']:+.4f} mean_pred={m_base['mean_pred']:.3f}")
    print(f"counterfactual: rmse={m_cf['rmse']:.4f} mae={m_cf['mae']:.4f} "
          f"bias={m_cf['bias']:+.4f} mean_pred={m_cf['mean_pred']:.3f}")
    print(f"mean_true={m_cf['mean_true']:.3f}  n={m_cf['n']}  elapsed={elapsed/60:.1f} min")

    print("\n  baseline RMSE by popularity bucket:")
    for b, s in sorted(m_base["by_popularity"].items()):
        print(f"    {b} (n={s['n']}, mean_pop={s['mean_pop']:.1f}): "
              f"rmse={s['rmse']:.4f} bias={s['bias']:+.4f}")
    print("  counterfactual RMSE by popularity bucket:")
    for b, s in sorted(m_cf["by_popularity"].items()):
        print(f"    {b} (n={s['n']}, mean_pop={s['mean_pop']:.1f}): "
              f"rmse={s['rmse']:.4f} bias={s['bias']:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"{tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"baseline": m_base, "counterfactual": m_cf,
                   "config": {"n": n, "k": k, "tips_window_days": TIPS_WINDOW_DAYS}}, f, indent=2)
    print(f"\nSaved: {out_dir}")
    return df, m_base, m_cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--k", type=int, default=K_EXPLANATIONS)
    ap.add_argument("--concurrency", type=int, default=LLM_CONCURRENCY)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", type=str, default="shake")
    args = ap.parse_args()
    run(n=args.n, k=args.k, concurrency=args.concurrency, seed=args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
