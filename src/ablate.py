"""Ablation: sweep TIPS_ALPHA and a no-intervention control on the same
sampled pairs used by run.py. Reuses the LLM cache aggressively: abduction
and intervention scoring outputs are already cached from the main run,
so only the prediction step (whose prompt depends on the chosen explanation)
may incur fresh LLM calls."""
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


def _select_no_intervention(scored_hypotheses):
    """Control: pick the highest-plausibility hypothesis, no pop penalty."""
    best = max(scored_hypotheses, key=lambda h: h["plausibility"])
    return {**best, "tips_pop_score": None, "causal_score": best["plausibility"]}


def predict_one(user_row, history_df, target_movie, movies_lookup,
                prop_table, window_secs, bucket_max, pop_quant,
                target_mid, target_ts, k, alpha, mode):
    # abduction + intervention scoring hit cache
    hyps = generate_candidates(user_row, history_df, target_movie, movies_lookup, k=k)
    scored = score_hypotheses(user_row, history_df, target_movie, movies_lookup, hyps)
    if mode == "no_intervention":
        best = _select_no_intervention(scored)
    else:
        best, _ = select_debiased(
            scored, target_mid, target_ts,
            prop_table, window_secs, bucket_max, pop_quant,
            alpha=alpha,
        )
    r, _ = predict_rating(user_row, target_movie, best["text"])
    return {
        "pred": r,
        "chosen_id": best["id"],
        "chosen_plausibility": best.get("plausibility"),
        "chosen_pop_reliance": best.get("popularity_reliance"),
        "tips_pop_score": best.get("tips_pop_score"),
    }


def metrics(y, p, pop, n_buckets=POPULARITY_BUCKETS):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pop = np.asarray(pop, dtype=float)
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
            "n": int(m.sum()),
            "mean_pop": float(pop[m].mean()),
            "rmse": float(np.sqrt(np.mean((y[m] - p[m]) ** 2))),
            "bias": float(np.mean(p[m] - y[m])),
        }
    out["by_popularity"] = buckets
    return out


def run(n, k, concurrency, seed, alphas, tag):
    t0 = time.time()
    movies = load_movies()
    users = load_users()
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)
    pop_quant = popularity_quantile(pop).to_dict()
    prop_table, window_secs, bucket_max = build_tips_table(train, window_days=TIPS_WINDOW_DAYS)

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    sample = stratified_subsample(test, pop, n=n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)

    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}
    users_lookup = {int(uid): users.loc[uid].to_dict() for uid in users.index}
    hist_cache = {int(uid): user_history(train, int(uid)) for uid in sample["user_id"].unique()}
    rows = sample.reset_index(drop=True).to_dict("records")
    print(f"Sampled {len(rows)} pairs; sweeping {len(alphas)} settings")

    settings = [("no_intervention", None)] + [("alpha", a) for a in alphas]
    results = {}
    y_true = np.array([r["rating"] for r in rows])
    pops = np.array([r["pop_count"] for r in rows])

    for mode, alpha in settings:
        label = "no_intervention" if mode == "no_intervention" else f"alpha={alpha:.2f}"
        def work(row, _m=mode, _a=alpha):
            u = users_lookup[int(row["user_id"])]
            m = movies_lookup[int(row["movie_id"])]
            h = hist_cache[int(row["user_id"])]
            try:
                return predict_one(
                    u, h, m, movies_lookup, prop_table, window_secs,
                    bucket_max, pop_quant,
                    row["movie_id"], row["timestamp"], k, _a, _m,
                )
            except Exception as e:
                return {"pred": 3.0, "err": str(e)[:120]}

        with tqdm(total=len(rows), desc=label) as pbar:
            outs = batch_map(work, rows, concurrency=concurrency, progress=pbar)

        preds = np.array([o["pred"] for o in outs])
        m = metrics(y_true, preds, pops)
        results[label] = {"metrics": m, "preds": preds.tolist()}
        print(f"  {label}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:+.4f} mean_pred={m['mean_pred']:.3f}")

    out_dir = Path(OUTPUT_DIR) / f"ablate_{tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = []
    for label, r in results.items():
        m = r["metrics"]
        table.append({
            "setting": label,
            "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
            "mean_pred": m["mean_pred"],
            **{f"{bk}_bias": bv["bias"] for bk, bv in m["by_popularity"].items()},
            **{f"{bk}_rmse": bv["rmse"] for bk, bv in m["by_popularity"].items()},
        })
    table_df = pd.DataFrame(table)
    table_df.to_csv(out_dir / "ablation_table.csv", index=False)
    with open(out_dir / "ablation.json", "w") as f:
        json.dump({
            "settings": [{"label": k, "metrics": v["metrics"]} for k, v in results.items()],
            "config": {"n": n, "k": k, "seed": seed, "alphas": alphas},
        }, f, indent=2)
    print(f"\nSaved: {out_dir}")
    print("\n=== Ablation summary (overall) ===")
    print(table_df[["setting", "rmse", "mae", "bias", "mean_pred"]].to_string(index=False))

    # Popularity-stratified bias table
    print("\n=== Bias by popularity bucket ===")
    rows_out = []
    for label, r in results.items():
        mm = r["metrics"]["by_popularity"]
        rows_out.append({"setting": label, **{bk: f"{bv['bias']:+.3f}" for bk, bv in mm.items()}})
    print(pd.DataFrame(rows_out).to_string(index=False))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--k", type=int, default=K_EXPLANATIONS)
    ap.add_argument("--concurrency", type=int, default=LLM_CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--tag", type=str, default="n500")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    run(args.n, args.k, args.concurrency, args.seed, alphas, args.tag)


if __name__ == "__main__":
    main()
