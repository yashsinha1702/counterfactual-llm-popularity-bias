"""Stacker v4: v2 baseline + fixed NeuMF ensemble + genre affinity features.

Minimal, focused experiment: take the PROVEN v2_best Ridge(α=100) at 0.8764,
and add ONLY the two things that showed genuine signal:
  1. Multi-seed NeuMF ensemble average (original architecture, 5 seeds)
  2. Genre affinity features (user's average rating for target movie's genres)

No kitchen-sink, no LightGBM — just test whether these two additions
can push below 0.8764.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from config import OUTPUT_DIR
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


def latest(pattern):
    m = sorted(glob.glob(str(pattern)))
    return m[-1] if m else None


def add_genre_affinity(df, train, movies):
    """Add per-pair genre affinity to df. Fully vectorized."""
    movies_df = movies.reset_index(drop=True) if movies.index.name == "movie_id" else movies
    mg = movies_df[["movie_id", "genres_list"]].dropna(subset=["genres_list"]).explode("genres_list")
    mg["genre_idx"] = mg["genres_list"].map(GENRE2IDX)
    mg = mg.dropna(subset=["genre_idx"]).copy()
    mg["genre_idx"] = mg["genre_idx"].astype(int)
    merged = train[["user_id", "movie_id", "rating"]].merge(
        mg[["movie_id", "genre_idx"]], on="movie_id", how="inner"
    )
    agg = merged.groupby(["user_id", "genre_idx"])["rating"].agg(["mean", "count"]).reset_index()

    # Build lookups
    ug_mean = {}
    ug_count = {}
    for uid, group in agg.groupby("user_id"):
        arr_m = np.zeros(len(ALL_GENRES), dtype=np.float32)
        arr_c = np.zeros(len(ALL_GENRES), dtype=np.float32)
        for _, r in group.iterrows():
            gi = int(r["genre_idx"])
            arr_m[gi] = float(r["mean"])
            arr_c[gi] = float(r["count"])
        ug_mean[int(uid)] = arr_m
        ug_count[int(uid)] = arr_c

    movie_genres = {}
    for mid_val in movies.index:
        row = movies.loc[mid_val]
        if isinstance(row.get("genres_list"), list):
            movie_genres[int(mid_val)] = [GENRE2IDX[g] for g in row["genres_list"] if g in GENRE2IDX]
        else:
            movie_genres[int(mid_val)] = []

    n = len(df)
    ga_mean = np.zeros(n)
    ga_std = np.zeros(n)
    ga_count = np.zeros(n)
    ga_range = np.zeros(n)

    uids = df["user_id"].astype(int).values
    mids = df["movie_id"].astype(int).values
    for idx in range(n):
        uid, mid = int(uids[idx]), int(mids[idx])
        glist = movie_genres.get(mid, [])
        if uid in ug_mean and glist:
            vals = [ug_mean[uid][g] for g in glist if ug_count.get(uid, np.zeros(1))[g] > 0] if uid in ug_count else []
            if not vals:
                vals = [ug_mean[uid][g] for g in glist if ug_mean[uid][g] > 0]
            if vals:
                ga_mean[idx] = np.mean(vals)
                ga_std[idx] = np.std(vals) if len(vals) > 1 else 0
                ga_range[idx] = max(vals) - min(vals)
                cnts = [ug_count[uid][g] for g in glist if ug_count[uid][g] > 0]
                ga_count[idx] = np.mean(cnts) if cnts else 0

    df = df.copy()
    df["genre_affinity_mean"] = ga_mean
    df["genre_affinity_std"] = ga_std
    df["genre_affinity_count"] = ga_count
    df["genre_affinity_range"] = ga_range
    return df


def cv_score(X, y, model_factory, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
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
    for tr, va in kf.split(X):
        dt = lgb.Dataset(X[tr], label=y[tr])
        dv = lgb.Dataset(X[va], label=y[va], reference=dt)
        m = lgb.train(params, dt, valid_sets=[dv], num_boost_round=1000,
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
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

    print("Building features...")
    from stacker import build_features
    df = build_features(tag=args.tag, model_suffixes=model_suffixes)
    y = df["rating"].astype(float).values

    # Add v4 NeuMF ensemble predictions
    ens_path = latest(OUTPUT_DIR / f"neumf_ens_v4_{args.tag}_*/predictions.csv")
    if ens_path:
        ens = pd.read_csv(ens_path)
        for c in [col for col in ens.columns if col.startswith("pred_NeuMF_") and "std" not in col]:
            d = ens[["user_id", "movie_id", c]].drop_duplicates(subset=["user_id", "movie_id"])
            df = df.merge(d, on=["user_id", "movie_id"], how="left")
            fill = df.get("pred_NeuMF", df["pred_ItemKNN"])
            df[c] = df[c].fillna(fill).clip(1.0, 5.0)
        if "pred_NeuMF_ens_std" in ens.columns:
            d = ens[["user_id", "movie_id", "pred_NeuMF_ens_std"]].drop_duplicates(subset=["user_id", "movie_id"])
            df = df.merge(d, on=["user_id", "movie_id"], how="left")
            df["pred_NeuMF_ens_std"] = df["pred_NeuMF_ens_std"].fillna(0)
        print(f"  v4 ensemble: {ens_path}")
    else:
        print("  WARNING: No neumf_ens_v4 predictions found!")

    # Add genre affinity features
    print("  Computing genre affinity...")
    ratings = load_ratings()
    train_r, _ = temporal_split(ratings)
    movies = load_movies()
    df = add_genre_affinity(df, train_r, movies)

    # Also compute prediction disagreement across ALL pred_ columns
    pred_cols = [c for c in df.columns if c.startswith("pred_") and df[c].notna().all()
                 and "std" not in c and "GlobalMean" not in c]
    if len(pred_cols) >= 3:
        pred_mat = df[pred_cols].values
        df["meta_pred_std"] = pred_mat.std(axis=1)
        df["meta_pred_range"] = pred_mat.max(axis=1) - pred_mat.min(axis=1)

    print(f"  n={len(df)}  columns={len(df.columns)}")

    # Feature sets
    # A) v2 baseline (for comparison)
    v2_base = [c for c in [
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

    # B) v2 + genre affinity only
    v4_genre = v2_base + [c for c in [
        "genre_affinity_mean", "genre_affinity_std",
        "genre_affinity_count", "genre_affinity_range",
    ] if c in df.columns]

    # C) v2 + NeuMF ensemble avg only (replace single NeuMF with ensemble)
    v4_ens = [c for c in v2_base if c != "pred_NeuMF"] + \
             [c for c in ["pred_NeuMF_ens_avg"] if c in df.columns]

    # D) v2 + ensemble avg + genre affinity
    v4_both = v4_ens + [c for c in [
        "genre_affinity_mean", "genre_affinity_std",
        "genre_affinity_count", "genre_affinity_range",
    ] if c in df.columns]

    # E) v2 + all individual NeuMF seeds + genre affinity + meta
    v4_full = v2_base + [c for c in sorted(df.columns)
                         if c.startswith("pred_NeuMF_") and "std" not in c and c in df.columns] + \
              [c for c in [
                  "genre_affinity_mean", "genre_affinity_std",
                  "genre_affinity_count", "genre_affinity_range",
                  "meta_pred_std", "meta_pred_range",
                  "pred_NeuMF_ens_std",
              ] if c in df.columns]
    # Deduplicate
    v4_full = list(dict.fromkeys(v4_full))

    # F) Minimal: just the strongest CF models + genre + ensemble
    v4_minimal = [c for c in [
        "pred_NeuMF_ens_avg", "pred_NeuMF", "pred_BPRMF",
        "pred_ItemKNN", "pred_SVD", "pred_Baseline", "pred_UserMean",
        "pred_cfv2_cal", "pred_diverse",
        "pred_cfv2_retr_14b", "pred_cfv2_retr_72b",
        "item_mean", "user_mean_hist", "log_pop",
        "genre_affinity_mean",
        "cfv2_delta_vs_knn", "knn_vs_usermean",
    ] if c in df.columns]

    FSETS = {
        "v2_base": v2_base,
        "v4_genre": v4_genre,
        "v4_ens": v4_ens,
        "v4_both": v4_both,
        "v4_full": v4_full,
        "v4_minimal": v4_minimal,
    }

    results = []
    for fs_name, feats in FSETS.items():
        if not feats:
            continue
        X = np.nan_to_num(df[feats].values.astype(float), nan=0.0)
        for alpha in [10.0, 100.0, 1000.0]:
            rmse, mae, bias, _ = cv_score(X, y, lambda a=alpha: Ridge(alpha=a, random_state=0))
            results.append({"features": fs_name, "model": f"Ridge(α={alpha})",
                           "rmse": round(rmse, 4), "mae": round(mae, 4),
                           "bias": round(bias, 4), "n_feats": len(feats)})
        rmse, mae, bias, oof_r = cv_score(X, y,
            lambda: RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3))
        results.append({"features": fs_name, "model": "RidgeCV",
                        "rmse": round(rmse, 4), "mae": round(mae, 4),
                        "bias": round(bias, 4), "n_feats": len(feats)})

    # LightGBM if available
    if HAVE_LGB:
        lgb_params = {
            "objective": "regression", "metric": "rmse", "verbosity": -1,
            "max_depth": 3, "num_leaves": 7,
            "learning_rate": 0.015, "feature_fraction": 0.6,
            "bagging_fraction": 0.7, "bagging_freq": 5,
            "min_child_samples": 30, "lambda_l2": 3.0, "lambda_l1": 0.5,
            "seed": 0,
        }
        for fs_name, feats in FSETS.items():
            if not feats:
                continue
            X = np.nan_to_num(df[feats].values.astype(float), nan=0.0)
            rmse, mae, bias, oof_l = cv_score_lgb(X, y, lgb_params)
            results.append({"features": fs_name, "model": "LGB_cons",
                           "rmse": round(rmse, 4), "mae": round(mae, 4),
                           "bias": round(bias, 4), "n_feats": len(feats)})
            # Blend
            _, _, _, oof_r = cv_score(X, y,
                lambda: RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3))
            for w in [0.7, 0.8, 0.9]:
                bl = np.clip(w * oof_r + (1 - w) * oof_l, 1.0, 5.0)
                rmse_bl = float(np.sqrt(np.mean((y - bl) ** 2)))
                results.append({"features": fs_name, "model": f"Blend(R={w:.1f}+L={1-w:.1f})",
                               "rmse": round(rmse_bl, 4), "n_feats": len(feats)})

    rdf = pd.DataFrame(results).sort_values("rmse")
    print("\n=== Stacker v4 Leaderboard ===")
    print(rdf.head(30).to_string(index=False))

    # Coefficient inspection for best feature set
    for candidate_key in ("v4_both", "v4_genre", "v4_ens", "v4_full", "v4_minimal", "v2_base"):
        feats = FSETS.get(candidate_key, [])
        if len(feats) >= 5:
            break
    X = np.nan_to_num(df[feats].values.astype(float), nan=0.0)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3)
    ridge.fit(X, y)
    coef_tbl = pd.DataFrame({"feature": feats, "coef": np.round(ridge.coef_, 4)
    }).sort_values("coef", key=abs, ascending=False)
    print(f"\n=== Coefficients: {candidate_key} ({len(feats)} feats, α={ridge.alpha_:.2f}) ===")
    print(coef_tbl.to_string(index=False))

    out_dir = Path(OUTPUT_DIR) / "stacker_v4"
    out_dir.mkdir(exist_ok=True)
    rdf.to_csv(out_dir / "cv_scores.csv", index=False)
    coef_tbl.to_csv(out_dir / "ridge_coefs.csv", index=False)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
