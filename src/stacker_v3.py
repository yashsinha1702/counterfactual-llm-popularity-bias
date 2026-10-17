"""Enhanced stacker v3: LightGBM + rich meta-features + all v3 model predictions.

Key additions over stacker.py:
  1. LightGBM with proper regularization (replaces sklearn GBT)
  2. Rich meta-features: prediction disagreement, genre affinity, user activity
  3. NeuMF ensemble (multi-seed) predictions
  4. DeepFM predictions
  5. Residual NeuMF predictions
  6. Blended Ridge + LightGBM output
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from config import OUTPUT_DIR, POPULARITY_BUCKETS
from data import load_ratings, load_movies, temporal_split, item_popularity

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE2IDX = {g: i for i, g in enumerate(ALL_GENRES)}

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False
    print("WARNING: lightgbm not installed, skipping LightGBM stacker")


def latest(pattern):
    m = sorted(glob.glob(str(pattern)))
    return m[-1] if m else None


def build_genre_features(df, train, movies):
    """Compute per-pair genre affinity features."""
    # Build user-genre rating matrix
    movies_df = movies.reset_index(drop=True) if movies.index.name == "movie_id" else movies
    merged = train.merge(movies_df[["movie_id", "genres_list"]], on="movie_id", how="left")
    records = []
    for _, row in merged.iterrows():
        if isinstance(row["genres_list"], list):
            for g in row["genres_list"]:
                if g in GENRE2IDX:
                    records.append((int(row["user_id"]), GENRE2IDX[g], float(row["rating"])))
    aff_df = pd.DataFrame(records, columns=["user_id", "genre_idx", "rating"])
    user_genre_mean = aff_df.groupby(["user_id", "genre_idx"])["rating"].mean().reset_index()
    user_genre_count = aff_df.groupby(["user_id", "genre_idx"])["rating"].count().reset_index()
    user_genre_count.columns = ["user_id", "genre_idx", "count"]

    # Build lookup dicts
    ug_mean = {}
    ug_count = {}
    for _, r in user_genre_mean.iterrows():
        ug_mean.setdefault(int(r["user_id"]), {})[int(r["genre_idx"])] = float(r["rating"])
    for _, r in user_genre_count.iterrows():
        ug_count.setdefault(int(r["user_id"]), {})[int(r["genre_idx"])] = int(r["count"])

    # Build movie genre lookup
    movie_genres = {}
    for mid, row in movies.iterrows():
        if isinstance(row.get("genres_list"), list):
            movie_genres[int(mid)] = [GENRE2IDX[g] for g in row["genres_list"] if g in GENRE2IDX]
        else:
            movie_genres[int(mid)] = []

    # Compute features per test pair
    ga_mean = np.zeros(len(df))
    ga_std = np.zeros(len(df))
    ga_count = np.zeros(len(df))
    ga_min = np.zeros(len(df))
    ga_max = np.zeros(len(df))
    n_matching_genres = np.zeros(len(df))

    for idx, (_, row) in enumerate(df.iterrows()):
        uid, mid = int(row["user_id"]), int(row["movie_id"])
        glist = movie_genres.get(mid, [])
        if uid in ug_mean and glist:
            vals = [ug_mean[uid].get(g, 0) for g in glist]
            counts = [ug_count.get(uid, {}).get(g, 0) for g in glist]
            active = [v for v, c in zip(vals, counts) if c > 0]
            if active:
                ga_mean[idx] = np.mean(active)
                ga_std[idx] = np.std(active) if len(active) > 1 else 0
                ga_min[idx] = np.min(active)
                ga_max[idx] = np.max(active)
                ga_count[idx] = np.mean([c for c in counts if c > 0])
                n_matching_genres[idx] = len(active)

    df = df.copy()
    df["genre_affinity_mean"] = ga_mean
    df["genre_affinity_std"] = ga_std
    df["genre_affinity_min"] = ga_min
    df["genre_affinity_max"] = ga_max
    df["genre_affinity_count"] = ga_count
    df["n_matching_genres"] = n_matching_genres
    return df


def build_features_v3(tag="n2000", model_suffixes=None):
    """Build the full feature matrix including all v3 additions."""
    # Load base features from existing stacker infrastructure
    from stacker import build_features
    df = build_features(tag=tag, model_suffixes=model_suffixes)

    # Load v3 predictions
    # NeuMF ensemble
    ens_path = latest(OUTPUT_DIR / f"neumf_ensemble_{tag}_*/predictions.csv")
    if ens_path:
        ens = pd.read_csv(ens_path)
        ens_cols = [c for c in ens.columns if c.startswith("pred_NeuMF_") and c != "pred_NeuMF_std"]
        for c in ens_cols:
            d = ens[["user_id", "movie_id", c]].drop_duplicates(subset=["user_id", "movie_id"])
            df = df.merge(d, on=["user_id", "movie_id"], how="left")
            df[c] = df[c].fillna(df.get("pred_NeuMF", df["pred_ItemKNN"])).clip(1.0, 5.0)
        # Also get std (disagreement signal, not a rating prediction — don't clip)
        if "pred_NeuMF_std" in ens.columns:
            d = ens[["user_id", "movie_id", "pred_NeuMF_std"]].drop_duplicates(subset=["user_id", "movie_id"])
            df = df.merge(d, on=["user_id", "movie_id"], how="left")
            df["pred_NeuMF_std"] = df["pred_NeuMF_std"].fillna(0)
        print(f"  neumf_ensemble: {ens_path} ({len(ens_cols)} models)")

    # DeepFM
    dfm_path = latest(OUTPUT_DIR / f"deepfm_{tag}_*/predictions.csv")
    if dfm_path:
        dfm = pd.read_csv(dfm_path)
        dfm_cols = [c for c in dfm.columns if c.startswith("pred_DeepFM_")]
        for c in dfm_cols:
            d = dfm[["user_id", "movie_id", c]].drop_duplicates(subset=["user_id", "movie_id"])
            df = df.merge(d, on=["user_id", "movie_id"], how="left")
            df[c] = df[c].fillna(df.get("pred_NeuMF", df["pred_ItemKNN"])).clip(1.0, 5.0)
        print(f"  deepfm: {dfm_path} ({len(dfm_cols)} models)")

    # Residual NeuMF
    res_path = latest(OUTPUT_DIR / f"residual_neumf_{tag}_*/predictions.csv")
    if res_path:
        res = pd.read_csv(res_path)
        res_cols = [c for c in res.columns if c.startswith("pred_ResidualNeuMF_")]
        for c in res_cols:
            d = res[["user_id", "movie_id", c]].drop_duplicates(subset=["user_id", "movie_id"])
            df = df.merge(d, on=["user_id", "movie_id"], how="left")
            df[c] = df[c].fillna(df.get("pred_NeuMF", df["pred_ItemKNN"])).clip(1.0, 5.0)
        print(f"  residual_neumf: {res_path} ({len(res_cols)} models)")

    # Genre affinity features
    print("  Computing genre affinity features...")
    ratings = load_ratings()
    train_r, _ = temporal_split(ratings)
    movies = load_movies()
    df = build_genre_features(df, train_r, movies)

    # Prediction disagreement features (meta-features for the stacker)
    pred_cols = [c for c in df.columns if c.startswith("pred_") and
                 c not in ("pred_GlobalMean",) and df[c].notna().all()]
    if len(pred_cols) >= 3:
        pred_matrix = df[pred_cols].values
        df["meta_pred_std"] = pred_matrix.std(axis=1)
        df["meta_pred_range"] = pred_matrix.max(axis=1) - pred_matrix.min(axis=1)
        df["meta_pred_median"] = np.median(pred_matrix, axis=1)
        df["meta_pred_iqr"] = np.percentile(pred_matrix, 75, axis=1) - np.percentile(pred_matrix, 25, axis=1)
        print(f"  Meta-features from {len(pred_cols)} prediction columns")

    return df


def cv_score(X, y, model_factory, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X[val_idx])
    oof = np.clip(oof, 1.0, 5.0)
    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    mae = float(np.mean(np.abs(y - oof)))
    bias = float(np.mean(oof - y))
    return rmse, mae, bias, oof


def cv_score_lgb(X, y, params, n_splits=5, seed=0):
    if not HAVE_LGB:
        return 999, 999, 0, np.full_like(y, 3.5)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        d_train = lgb.Dataset(X[train_idx], label=y[train_idx])
        d_val = lgb.Dataset(X[val_idx], label=y[val_idx], reference=d_train)
        model = lgb.train(
            params, d_train,
            valid_sets=[d_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict(X[val_idx])
    oof = np.clip(oof, 1.0, 5.0)
    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    mae = float(np.mean(np.abs(y - oof)))
    bias = float(np.mean(oof - y))
    return rmse, mae, bias, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="n2000")
    ap.add_argument("--model_suffixes", type=str, default=None)
    args = ap.parse_args()
    model_suffixes = args.model_suffixes.split(",") if args.model_suffixes else None

    print(f"Building v3 features (tag={args.tag})...")
    df = build_features_v3(tag=args.tag, model_suffixes=model_suffixes)
    y = df["rating"].astype(float).values
    print(f"  n={len(df)}  mean_rating={y.mean():.3f}")

    # Define feature sets
    # v2 baseline (best previous)
    v2_best = [c for c in [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline", "pred_UserMean",
        "pred_NeuMF", "pred_LightGCN", "pred_LightGCN_llminit",
        "pred_BPRMF", "pred_TwoTower",
        "pred_cfv2_cal", "pred_diverse",
        "pred_diverse_retr_14b", "pred_diverse_retr_72b",
        "pred_cfv2_retr_14b", "pred_cfv2_retr_72b",
        "pred_soft_dist",
        "soft_p1", "soft_p2", "soft_p3", "soft_p4", "soft_p5",
        "pred_PreferenceCoT_cal", "pred_Analogical_cal",
        "pred_ConfidenceRouter",
        "item_mean", "user_mean_hist", "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean",
    ] if c in df.columns]

    # v3 NeuMF ensemble features
    neumf_ens_cols = sorted([c for c in df.columns if c.startswith("pred_NeuMF_") and c != "pred_NeuMF_std"])
    deepfm_cols = sorted([c for c in df.columns if c.startswith("pred_DeepFM_")])
    residual_cols = sorted([c for c in df.columns if c.startswith("pred_ResidualNeuMF_")])

    # v3: focused ensemble (best models only, no noise)
    v3_focused = [c for c in [
        "pred_NeuMF", "pred_BPRMF", "pred_ItemKNN", "pred_SVD",
        "pred_Baseline", "pred_UserMean",
        "pred_NeuMF_avg_all", "pred_NeuMF_avg_d64", "pred_NeuMF_avg_d128",
        "pred_DeepFM_avg", "pred_ResidualNeuMF_avg",
        "pred_cfv2_cal", "pred_diverse",
        "pred_cfv2_retr_14b", "pred_cfv2_retr_72b",
        "item_mean", "user_mean_hist", "log_pop", "user_rating_count",
        "genre_affinity_mean", "genre_affinity_std", "genre_affinity_count",
        "n_matching_genres",
        "meta_pred_std", "meta_pred_range",
        "cfv2_delta_vs_knn", "knn_vs_usermean",
    ] if c in df.columns]

    # v3: kitchen sink (everything)
    v3_all = sorted(list(set(
        v2_best + neumf_ens_cols + deepfm_cols + residual_cols +
        [c for c in [
            "pred_NeuMF_std",
            "genre_affinity_mean", "genre_affinity_std", "genre_affinity_min",
            "genre_affinity_max", "genre_affinity_count", "n_matching_genres",
            "meta_pred_std", "meta_pred_range", "meta_pred_median", "meta_pred_iqr",
        ] if c in df.columns]
    )))

    # v3: CF-strong with ensemble
    v3_cf_ens = [c for c in [
        "pred_NeuMF", "pred_NeuMF_avg_all", "pred_NeuMF_avg_d64", "pred_NeuMF_avg_d128",
        "pred_BPRMF", "pred_ItemKNN", "pred_SVD", "pred_Baseline", "pred_UserMean",
        "pred_DeepFM_avg", "pred_ResidualNeuMF_avg",
        "item_mean", "user_mean_hist", "log_pop",
        "genre_affinity_mean", "genre_affinity_count",
        "meta_pred_std",
    ] if c in df.columns]

    FEATURE_SETS = {
        "v2_best": v2_best,
        "v3_focused": v3_focused,
        "v3_cf_ensemble": v3_cf_ens,
        "v3_all": v3_all,
    }

    results = []

    # Ridge stacker
    print("\n=== Ridge Stacker (5-fold CV) ===")
    for fs_name, feats in FEATURE_SETS.items():
        if not feats:
            continue
        X = df[feats].values.astype(float)
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        for alpha in [1.0, 10.0, 100.0]:
            rmse, mae, bias, oof = cv_score(X, y, lambda a=alpha: Ridge(alpha=a, random_state=0))
            results.append({"features": fs_name, "model": f"Ridge(α={alpha})",
                           "rmse": round(rmse, 4), "mae": round(mae, 4),
                           "bias": round(bias, 4), "n_feats": len(feats)})
        rmse, mae, bias, oof_ridge = cv_score(X, y,
            lambda: RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3))
        results.append({"features": fs_name, "model": "RidgeCV",
                        "rmse": round(rmse, 4), "mae": round(mae, 4),
                        "bias": round(bias, 4), "n_feats": len(feats)})

    # LightGBM stacker
    if HAVE_LGB:
        print("\n=== LightGBM Stacker (5-fold CV) ===")
        lgb_configs = [
            {"name": "lgb_default", "params": {
                "objective": "regression", "metric": "rmse", "verbosity": -1,
                "n_estimators": 500, "max_depth": 4, "num_leaves": 15,
                "learning_rate": 0.03, "feature_fraction": 0.7,
                "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_child_samples": 20, "lambda_l2": 1.0, "lambda_l1": 0.1,
                "seed": 0,
            }},
            {"name": "lgb_conservative", "params": {
                "objective": "regression", "metric": "rmse", "verbosity": -1,
                "n_estimators": 800, "max_depth": 3, "num_leaves": 7,
                "learning_rate": 0.015, "feature_fraction": 0.6,
                "bagging_fraction": 0.7, "bagging_freq": 5,
                "min_child_samples": 30, "lambda_l2": 3.0, "lambda_l1": 0.5,
                "seed": 0,
            }},
            {"name": "lgb_deep", "params": {
                "objective": "regression", "metric": "rmse", "verbosity": -1,
                "n_estimators": 600, "max_depth": 5, "num_leaves": 20,
                "learning_rate": 0.02, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 3,
                "min_child_samples": 15, "lambda_l2": 0.5, "lambda_l1": 0.05,
                "seed": 0,
            }},
        ]
        best_lgb_oof = {}
        for fs_name, feats in FEATURE_SETS.items():
            if not feats:
                continue
            X = df[feats].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            for cfg in lgb_configs:
                rmse, mae, bias, oof = cv_score_lgb(X, y, cfg["params"])
                results.append({"features": fs_name, "model": cfg["name"],
                               "rmse": round(rmse, 4), "mae": round(mae, 4),
                               "bias": round(bias, 4), "n_feats": len(feats)})
                key = (fs_name, cfg["name"])
                if key not in best_lgb_oof or rmse < best_lgb_oof[key][0]:
                    best_lgb_oof[key] = (rmse, oof)

    # Blended Ridge + LightGBM
    if HAVE_LGB:
        print("\n=== Blended Ridge + LightGBM ===")
        for fs_name, feats in FEATURE_SETS.items():
            if not feats:
                continue
            X = df[feats].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            _, _, _, oof_r = cv_score(X, y, lambda: RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3))
            _, _, _, oof_l = cv_score_lgb(X, y, lgb_configs[0]["params"])
            for alpha in [0.3, 0.5, 0.7]:
                blended = np.clip(alpha * oof_r + (1 - alpha) * oof_l, 1.0, 5.0)
                rmse = float(np.sqrt(np.mean((y - blended) ** 2)))
                mae = float(np.mean(np.abs(y - blended)))
                bias = float(np.mean(blended - y))
                results.append({"features": fs_name, "model": f"Blend(Ridge={alpha:.1f}+LGB={1-alpha:.1f})",
                               "rmse": round(rmse, 4), "mae": round(mae, 4),
                               "bias": round(bias, 4), "n_feats": len(feats)})

    # Print results
    rdf = pd.DataFrame(results).sort_values("rmse")
    print("\n=== Full Leaderboard ===")
    print(rdf.head(40).to_string(index=False))

    # Coefficient inspection for best Ridge
    print("\n=== Best Ridge: coefficient inspection ===")
    for candidate_key in ("v3_focused", "v3_cf_ensemble", "v3_all", "v2_best"):
        feats = FEATURE_SETS.get(candidate_key, [])
        if len(feats) >= 5:
            break
    X = df[feats].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3)
    ridge.fit(X, y)
    coef_tbl = pd.DataFrame({
        "feature": feats, "coef": np.round(ridge.coef_, 4),
    }).sort_values("coef", key=abs, ascending=False)
    print(f"  Feature set: {candidate_key} ({len(feats)} feats)")
    print(f"  Alpha: {ridge.alpha_:.4f}  Intercept: {ridge.intercept_:.4f}")
    print(coef_tbl.to_string(index=False))

    # Feature importance from LightGBM
    if HAVE_LGB:
        print("\n=== LightGBM Feature Importance ===")
        feats = FEATURE_SETS.get("v3_focused", FEATURE_SETS.get("v3_all", []))
        if feats:
            X = df[feats].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            d_full = lgb.Dataset(X, label=y)
            model_full = lgb.train(lgb_configs[0]["params"], d_full, num_boost_round=300)
            imp = pd.DataFrame({
                "feature": feats,
                "importance": model_full.feature_importance("gain"),
            }).sort_values("importance", ascending=False)
            print(imp.head(20).to_string(index=False))

    # Save
    out_dir = Path(OUTPUT_DIR) / "stacker_v3"
    out_dir.mkdir(exist_ok=True)
    rdf.to_csv(out_dir / "cv_scores.csv", index=False)
    coef_tbl.to_csv(out_dir / "ridge_coefs.csv", index=False)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
