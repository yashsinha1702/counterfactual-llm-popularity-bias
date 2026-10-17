"""CFv2 causal-delta pipeline with retrieval-augmented abduction.
Identical to run_cf_v2.py except user_history is selected via content
similarity to the target item. This gives the LLM semantically relevant
evidence for its counterfactual reasoning."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    OUTPUT_DIR, SUBSAMPLE_N, POPULARITY_BUCKETS, TIPS_WINDOW_DAYS,
    K_EXPLANATIONS, LLM_CONCURRENCY, TIPS_ALPHA,
)
from data import (
    load_movies, load_users, load_ratings,
    temporal_split, item_popularity, popularity_quantile, build_tips_table,
    stratified_subsample,
)
from llm import batch_map
from abduction import generate_candidates
from intervention import score_hypotheses, select_debiased
from prediction_v2 import predict_rating_delta
from cf_baseline import (
    fit_baselines_als, fit_itemknn, predict_itemknn,
)
from retrieval_utils import user_history_retrieved


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
    ap.add_argument("--k", type=int, default=K_EXPLANATIONS)
    ap.add_argument("--concurrency", type=int, default=LLM_CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=TIPS_ALPHA)
    ap.add_argument("--tag", type=str, default="cfv2_retr")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--k_retrieve", type=int, default=15)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading...")
    movies = load_movies()
    users = load_users()
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    pop = item_popularity(train)
    pop_quant = popularity_quantile(pop).to_dict()
    prop_table, window_secs, bucket_max = build_tips_table(train, window_days=TIPS_WINDOW_DAYS)

    print("Fitting ItemKNN CF model on train...")
    user_idx = {int(u): i for i, u in enumerate(sorted(train["user_id"].unique()))}
    item_idx = {int(m): i for i, m in enumerate(sorted(train["movie_id"].unique()))}
    mu = float(train["rating"].mean())
    user_mean_map = train.groupby("user_id")["rating"].mean().to_dict()
    b_u, b_i = fit_baselines_als(train, user_idx, item_idx, mu)
    sim_mat, R_sparse = fit_itemknn(train, user_idx, item_idx, mu, b_u, b_i)

    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)
    print(f"  n_test={len(sample)}  k_retrieve={args.k_retrieve}")

    cf_ref = predict_itemknn(sample["user_id"].values, sample["movie_id"].values,
                             user_idx, item_idx, mu, b_u, b_i, sim_mat, R_sparse, k=40)
    cf_ref = np.clip(cf_ref, 1.0, 5.0)

    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}
    users_lookup = {int(uid): users.loc[uid].to_dict() for uid in users.index}
    rows = sample.reset_index(drop=True).to_dict("records")

    def work(idx_row):
        i, row = idx_row
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = user_history_retrieved(train, int(row["user_id"]),
                                   int(row["movie_id"]),
                                   max_items=args.k_retrieve)
        ref = float(cf_ref[i])
        um = float(user_mean_map.get(int(row["user_id"]), mu))
        try:
            hyps = generate_candidates(u, h, m, movies_lookup, k=args.k)
            scored = score_hypotheses(u, h, m, movies_lookup, hyps)
            best, _ = select_debiased(
                scored, row["movie_id"], row["timestamp"],
                prop_table, window_secs, bucket_max, pop_quant,
                alpha=args.alpha,
            )
            r, delta, direction, brief = predict_rating_delta(
                u, m, best["text"], ref, user_mean_hint=um
            )
        except Exception as e:
            return {"pred": ref, "delta": 0.0, "direction": "err", "err": str(e)[:140]}
        return {"pred": r, "delta": delta, "direction": direction,
                "chosen_id": best["id"], "chosen_pop_reliance": best.get("popularity_reliance"),
                "chosen_plausibility": best.get("plausibility"),
                "tips_pop_score": best.get("tips_pop_score"),
                "brief": brief}

    print(f"\nRunning CFv2-retrieval (α={args.alpha}, n={len(rows)}, conc={args.concurrency})...")
    indexed = list(enumerate(rows))
    with tqdm(total=len(rows), desc="cfv2_retr") as pbar:
        outs = batch_map(work, indexed, concurrency=args.concurrency, progress=pbar)

    preds = np.clip(np.array([o["pred"] for o in outs], dtype=float), 1.0, 5.0)
    y = np.array([r["rating"] for r in rows], dtype=float)
    pops = np.array([r["pop_count"] for r in rows], dtype=float)

    m_raw = metrics(y, preds, pops)
    print(f"\nCFv2_retr (raw):     rmse={m_raw['rmse']:.4f} mae={m_raw['mae']:.4f} "
          f"bias={m_raw['bias']:+.4f} mean_pred={m_raw['mean_pred']:.3f}")
    m_knn = metrics(y, cf_ref, pops)
    print(f"ItemKNN reference:   rmse={m_knn['rmse']:.4f} mae={m_knn['mae']:.4f} "
          f"bias={m_knn['bias']:+.4f}")

    preds_cal = preds.copy()
    if args.calibrate:
        shift = -m_raw["bias"]
        preds_cal = np.clip(preds + shift, 1.0, 5.0)
        m_cal = metrics(y, preds_cal, pops)
        print(f"CFv2_retr (bias-cal, shift={shift:+.3f}): rmse={m_cal['rmse']:.4f} "
              f"mae={m_cal['mae']:.4f} bias={m_cal['bias']:+.4f}")

    out_dir = Path(OUTPUT_DIR) / f"cfv2_retrieval_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["cf_reference"] = cf_ref
    outdf["pred_cfv2_retr_raw"] = preds
    if args.calibrate:
        outdf["pred_cfv2_retr_cal"] = preds_cal
    for col, key in [("delta", "delta"), ("direction", "direction"),
                     ("chosen_pop_reliance", "chosen_pop_reliance"),
                     ("chosen_plausibility", "chosen_plausibility"),
                     ("tips_pop_score", "tips_pop_score"), ("brief", "brief")]:
        outdf[col] = [o.get(key) for o in outs]
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        blob = {"cfv2_retr_raw": m_raw, "itemknn_reference": m_knn, "config": vars(args)}
        if args.calibrate:
            blob["cfv2_retr_calibrated"] = m_cal
        json.dump(blob, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
