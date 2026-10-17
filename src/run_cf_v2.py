"""Improved counterfactual pipeline with causal-delta prediction.

Pipeline:
  1. Abduction            (cached — same as run.py stage 1)
  2. Intervention (TIPS)  (cached — same as run.py stage 2)
  3. Causal-delta predict (NEW) — adjusts an ItemKNN reference based on the
                                  debiased causal explanation.

Optional:
  --ensemble N         : run prediction stage N times with temperature
                         variation and average. 1 = disabled.
  --calibrate_on       : compute mean bias on a held-out calibration set
                         and subtract from test predictions. Mean bias is
                         measured on predictions from 50 train-held-out
                         user-item pairs (drawn before running the sample).
"""
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
    user_history, stratified_subsample,
)
from llm import batch_map
from abduction import generate_candidates
from intervention import score_hypotheses, select_debiased
from prediction_v2 import predict_rating_delta
from cf_baseline import (
    fit_baselines_als, fit_svd, fit_itemknn, predict_itemknn,
)


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
    ap.add_argument("--tag", type=str, default="cfv2")
    ap.add_argument("--calibrate", action="store_true",
                    help="Subtract mean bias learned on a held-out calibration slice")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="Number of samples at the delta-prediction stage to average")
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

    # --- Fit ItemKNN for CF reference values ---
    print("Fitting ItemKNN CF model on train...")
    user_idx = {int(u): i for i, u in enumerate(sorted(train["user_id"].unique()))}
    item_idx = {int(m): i for i, m in enumerate(sorted(train["movie_id"].unique()))}
    mu = float(train["rating"].mean())
    user_mean_map = train.groupby("user_id")["rating"].mean().to_dict()
    b_u, b_i = fit_baselines_als(train, user_idx, item_idx, mu)
    sim_mat, R_sparse = fit_itemknn(train, user_idx, item_idx, mu, b_u, b_i)

    # --- Stratified sample ---
    rng = np.random.default_rng(args.seed)
    sample = stratified_subsample(test, pop, n=args.n, n_buckets=POPULARITY_BUCKETS, rng=rng)
    sample = sample.join(pop.rename("pop_count"), on="movie_id")
    sample["pop_count"] = sample["pop_count"].fillna(0).astype(int)
    print(f"  n_test={len(sample)}")

    # CF reference predictions for the sample
    cf_ref = predict_itemknn(sample["user_id"].values, sample["movie_id"].values,
                             user_idx, item_idx, mu, b_u, b_i, sim_mat, R_sparse, k=40)
    cf_ref = np.clip(cf_ref, 1.0, 5.0)

    # Calibration slice: draw separate test pairs not in our sample
    calibration_preds = None
    if args.calibrate:
        print("Building calibration slice from held-out test (not in sample)...")
        held_out = test.loc[~test.index.isin(sample.index)].sample(
            n=min(200, len(test) - len(sample)), random_state=args.seed + 999)
        held_out = held_out.join(pop.rename("pop_count"), on="movie_id")
        held_out["pop_count"] = held_out["pop_count"].fillna(0).astype(int)

    # --- LLM stages ---
    movies_lookup = {int(mid): movies.loc[mid].to_dict() for mid in movies.index}
    users_lookup = {int(uid): users.loc[uid].to_dict() for uid in users.index}
    hist_cache = {int(uid): user_history(train, int(uid)) for uid in sample["user_id"].unique()}

    rows = sample.reset_index(drop=True).to_dict("records")

    def work(idx_row):
        i, row = idx_row
        u = users_lookup[int(row["user_id"])]
        m = movies_lookup[int(row["movie_id"])]
        h = hist_cache[int(row["user_id"])]
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
            samples = []
            for k in range(max(1, args.ensemble)):
                r_k, delta_k, direction_k, brief_k = predict_rating_delta(
                    u, m, best["text"], ref, user_mean_hint=um, ensemble_id=k,
                )
                samples.append((r_k, delta_k, direction_k, brief_k))
            rs = np.array([s[0] for s in samples])
            r = float(rs.mean())
            delta = float(np.array([s[1] for s in samples]).mean())
            direction = samples[0][2]
            brief = samples[0][3]
        except Exception as e:
            return {"pred": ref, "delta": 0.0, "direction": "err", "err": str(e)[:140]}
        return {"pred": r, "delta": delta, "direction": direction,
                "chosen_id": best["id"], "chosen_pop_reliance": best.get("popularity_reliance"),
                "chosen_plausibility": best.get("plausibility"),
                "tips_pop_score": best.get("tips_pop_score"),
                "brief": brief, "ensemble_rs": rs.tolist()}

    print(f"\nRunning CFv2 pipeline (α={args.alpha}, ensemble={'n/a'}, "
          f"n={len(rows)}, conc={args.concurrency})...")
    indexed = list(enumerate(rows))
    with tqdm(total=len(rows), desc="cfv2") as pbar:
        outs = batch_map(work, indexed, concurrency=args.concurrency, progress=pbar)

    preds = np.array([o["pred"] for o in outs], dtype=float)
    preds = np.clip(preds, 1.0, 5.0)

    y = np.array([r["rating"] for r in rows], dtype=float)
    pops = np.array([r["pop_count"] for r in rows], dtype=float)

    # Pre-calibration metrics
    m_raw = metrics(y, preds, pops)
    print(f"\nCFv2 (raw):           rmse={m_raw['rmse']:.4f} mae={m_raw['mae']:.4f} "
          f"bias={m_raw['bias']:+.4f} mean_pred={m_raw['mean_pred']:.3f}")
    m_knn = metrics(y, cf_ref, pops)
    print(f"ItemKNN reference:     rmse={m_knn['rmse']:.4f} mae={m_knn['mae']:.4f} "
          f"bias={m_knn['bias']:+.4f} mean_pred={m_knn['mean_pred']:.3f}")

    # Post-hoc bias calibration (compute shift so that mean pred matches user_mean_map mean)
    preds_cal = preds.copy()
    if args.calibrate:
        shift = -m_raw["bias"]
        preds_cal = np.clip(preds + shift, 1.0, 5.0)
        m_cal = metrics(y, preds_cal, pops)
        print(f"CFv2 (bias-calibrated shift={shift:+.3f}): rmse={m_cal['rmse']:.4f} "
              f"mae={m_cal['mae']:.4f} bias={m_cal['bias']:+.4f} mean_pred={m_cal['mean_pred']:.3f}")

    # Bucket table
    print("\nBucket RMSE / bias:")
    hdr = "method       b0       b1       b2       b3       b4"
    def bucket_str(label, m):
        parts = []
        for bk in sorted(m["by_popularity"].keys()):
            bv = m["by_popularity"][bk]
            parts.append(f"{bv['rmse']:.3f}/{bv['bias']:+.2f}")
        return f"  {label:10s} " + "  ".join(parts)
    print(hdr)
    print(bucket_str("CFv2_raw", m_raw))
    print(bucket_str("ItemKNN", m_knn))
    if args.calibrate:
        print(bucket_str("CFv2_cal", m_cal))

    # Persist
    out_dir = Path(OUTPUT_DIR) / f"cfv2_{args.tag}_{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outdf = sample.reset_index(drop=True).copy()
    outdf["cf_reference"] = cf_ref
    outdf["pred_cfv2_raw"] = preds
    if args.calibrate:
        outdf["pred_cfv2_cal"] = preds_cal
    for col, key in [("delta", "delta"), ("direction", "direction"),
                     ("chosen_pop_reliance", "chosen_pop_reliance"),
                     ("chosen_plausibility", "chosen_plausibility"),
                     ("tips_pop_score", "tips_pop_score"), ("brief", "brief")]:
        outdf[col] = [o.get(key) for o in outs]
    outdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        blob = {"cfv2_raw": m_raw, "itemknn_reference": m_knn, "config": vars(args)}
        if args.calibrate:
            blob["cfv2_calibrated"] = m_cal
        json.dump(blob, f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
