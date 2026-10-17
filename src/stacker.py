"""Stacked ensemble: fit a ridge regression on top of per-pair predictions
from every base method, with metadata features (popularity, user/item
means, counts). 5-fold CV reports an honest held-out RMSE — no method
sees its own target during fitting.

Feature set:
  Base-method predictions:
    - ItemKNN (residual cosine)
    - SVD (k=50 truncated SVD on residuals)
    - Baseline_CF (mu + b_u + b_i, ALS biases)
    - UserMean (user's historical mean)
    - GlobalMean (constant mu)
    - CFv2_cal (LLM causal-delta, bias-calibrated)
  Metadata:
    - log1p(pop_count)
    - user_bias = Baseline_CF - mu - item_bias_estimate  (extracted)
    - item_bias_estimate = ItemMean - mu
    - user_rating_count
  Interactions:
    - pop × CFv2_delta   where CFv2_delta = CFv2_cal - ItemKNN
    - pop × (ItemKNN - UserMean)  (item-specific vs user-default signal strength)
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

from config import OUTPUT_DIR
from data import load_ratings, temporal_split, item_popularity


def latest(pattern):
    m = sorted(glob.glob(str(pattern)))
    return m[-1] if m else None


def build_features(tag="n500", model_suffixes=None):
    """Merge all predictions + metadata into a single DataFrame.

    If model_suffixes is a list (e.g. ['14b', '72b']), also pull in
    diverse_retrieval_<tag>_<suffix>_* and cfv2_retrieval_<tag>_<suffix>_*
    and rename the prediction columns with the suffix so the stacker
    can learn independent weights per model.
    """
    cfv2_path = latest(OUTPUT_DIR / f"cfv2_{tag}_*/predictions.csv")
    cfb_path = latest(OUTPUT_DIR / f"cf_baseline_{tag}_*/predictions.csv")
    diverse_path = latest(OUTPUT_DIR / f"diverse_{tag}_*/predictions.csv")
    diverse_retr_path = latest(OUTPUT_DIR / f"diverse_retrieval_{tag}_*/predictions.csv")
    cfv2_retr_path = latest(OUTPUT_DIR / f"cfv2_retrieval_{tag}_*/predictions.csv")
    neumf_path = latest(OUTPUT_DIR / f"neumf_{tag}_*/predictions.csv")
    lightgcn_path = latest(OUTPUT_DIR / f"lightgcn_{tag}_*/predictions.csv")
    lightgcn_llm_path = latest(OUTPUT_DIR / f"lightgcn_llminit_{tag}_*/predictions.csv")
    bprmf_path = latest(OUTPUT_DIR / f"bprmf_{tag}_*/predictions.csv")
    twotower_path = latest(OUTPUT_DIR / f"twotower_{tag}_*/predictions.csv")
    soft_dist_path = latest(OUTPUT_DIR / f"soft_dist_{tag}_*/predictions.csv")
    cot_path = latest(OUTPUT_DIR / f"preference_cot_{tag}_*/predictions.csv")
    analog_path = latest(OUTPUT_DIR / f"analogical_{tag}_*/predictions.csv")
    router_path = latest(OUTPUT_DIR / f"confidence_router_{tag}_*/predictions.csv")
    if cfv2_path is None or cfb_path is None:
        raise FileNotFoundError(f"Need both cfv2_{tag}_* and cf_baseline_{tag}_* predictions")
    print(f"  cfv2:        {cfv2_path}")
    print(f"  cf_baseline: {cfb_path}")

    cfv2_all = pd.read_csv(cfv2_path)
    keep = ["user_id", "movie_id", "rating", "pop_count",
            "cf_reference", "pred_cfv2_raw", "pred_cfv2_cal",
            "delta", "direction", "chosen_pop_reliance",
            "chosen_plausibility", "tips_pop_score"]
    cfv2 = cfv2_all[[c for c in keep if c in cfv2_all.columns]]
    cfb = pd.read_csv(cfb_path)[
        ["user_id", "movie_id",
         "pred_GlobalMean", "pred_UserMean", "pred_Baseline",
         "pred_SVD", "pred_ItemKNN"]
    ]
    df = cfv2.merge(cfb, on=["user_id", "movie_id"], how="inner")

    # Merge in diverse-prompt LLM predictions (orthogonal LLM signal: not anchored on CF)
    if diverse_path is not None:
        div = pd.read_csv(diverse_path)
        if "pred_diverse" in div.columns:
            div = div[["user_id", "movie_id", "pred_diverse"]]
            df = df.merge(div, on=["user_id", "movie_id"], how="left")
            df["pred_diverse"] = df["pred_diverse"].fillna(df["pred_cfv2_cal"]).clip(1.0, 5.0)
            print(f"  diverse:            {diverse_path}")

    # Retrieval-augmented diverse predictions
    if diverse_retr_path is not None:
        d2 = pd.read_csv(diverse_retr_path)
        if "pred_diverse_retrieval" in d2.columns:
            d2 = d2[["user_id", "movie_id", "pred_diverse_retrieval"]]
            df = df.merge(d2, on=["user_id", "movie_id"], how="left")
            df["pred_diverse_retrieval"] = df["pred_diverse_retrieval"].fillna(df.get("pred_diverse", df["pred_cfv2_cal"])).clip(1.0, 5.0)
            print(f"  diverse_retrieval:  {diverse_retr_path}")

    # Retrieval-augmented CFv2
    if cfv2_retr_path is not None:
        c2 = pd.read_csv(cfv2_retr_path)
        col = "pred_cfv2_retr_cal" if "pred_cfv2_retr_cal" in c2.columns else "pred_cfv2_retr_raw"
        if col in c2.columns:
            c2 = c2[["user_id", "movie_id", col]].rename(columns={col: "pred_cfv2_retr"})
            df = df.merge(c2, on=["user_id", "movie_id"], how="left")
            df["pred_cfv2_retr"] = df["pred_cfv2_retr"].fillna(df["pred_cfv2_cal"]).clip(1.0, 5.0)
            print(f"  cfv2_retrieval:     {cfv2_retr_path}")

    # NeuMF predictions
    if neumf_path is not None:
        nm = pd.read_csv(neumf_path)
        if "pred_NeuMF" in nm.columns:
            nm = nm[["user_id", "movie_id", "pred_NeuMF"]]
            df = df.merge(nm, on=["user_id", "movie_id"], how="left")
            df["pred_NeuMF"] = df["pred_NeuMF"].fillna(df["pred_ItemKNN"]).clip(1.0, 5.0)
            print(f"  neumf:              {neumf_path}")

    # V2 additions: LightGCN, BPR-MF, TwoTower, and LLM soft-dist / CoT / analogical / router
    def _merge_pred(df, path, col, label, fill_col="pred_ItemKNN"):
        if path is None:
            return df
        d = pd.read_csv(path)
        if col not in d.columns:
            return df
        d = d[["user_id", "movie_id", col]].drop_duplicates(subset=["user_id", "movie_id"])
        df = df.merge(d, on=["user_id", "movie_id"], how="left")
        fill = df[fill_col] if fill_col in df.columns else 3.5
        df[col] = df[col].fillna(fill).clip(1.0, 5.0)
        print(f"  {label:20s}: {path}")
        return df

    df = _merge_pred(df, lightgcn_path, "pred_LightGCN", "lightgcn")
    df = _merge_pred(df, lightgcn_llm_path, "pred_LightGCN_llminit", "lightgcn_llminit")
    df = _merge_pred(df, bprmf_path, "pred_BPRMF", "bprmf")
    df = _merge_pred(df, twotower_path, "pred_TwoTower", "twotower")
    df = _merge_pred(df, soft_dist_path, "pred_soft_dist", "soft_dist")
    df = _merge_pred(df, cot_path, "pred_PreferenceCoT_cal", "preference_cot_cal")
    df = _merge_pred(df, cot_path, "pred_PreferenceCoT_raw", "preference_cot_raw")
    df = _merge_pred(df, analog_path, "pred_Analogical_cal", "analogical_cal")
    df = _merge_pred(df, analog_path, "pred_Analogical_raw", "analogical_raw")
    df = _merge_pred(df, router_path, "pred_ConfidenceRouter", "confidence_router")

    # Soft distribution probabilities (5 extra features)
    if soft_dist_path is not None:
        sd = pd.read_csv(soft_dist_path)
        for k in range(1, 6):
            col = f"soft_p{k}"
            if col in sd.columns:
                d = sd[["user_id", "movie_id", col]].drop_duplicates(subset=["user_id", "movie_id"])
                df = df.merge(d, on=["user_id", "movie_id"], how="left")
                df[col] = df[col].fillna(0.2)

    # Per-model retrieval variants: pred_diverse_retr_<suffix>, pred_cfv2_retr_<suffix>
    if model_suffixes:
        for suffix in model_suffixes:
            dr_path = latest(OUTPUT_DIR / f"diverse_retrieval_{tag}_{suffix}_*/predictions.csv")
            if dr_path is not None:
                d = pd.read_csv(dr_path)
                if "pred_diverse_retrieval" in d.columns:
                    d = d[["user_id", "movie_id", "pred_diverse_retrieval"]].rename(
                        columns={"pred_diverse_retrieval": f"pred_diverse_retr_{suffix}"})
                    df = df.merge(d, on=["user_id", "movie_id"], how="left")
                    fill = df.get("pred_diverse", df["pred_cfv2_cal"])
                    df[f"pred_diverse_retr_{suffix}"] = df[f"pred_diverse_retr_{suffix}"].fillna(fill).clip(1.0, 5.0)
                    print(f"  diverse_retr_{suffix}:   {dr_path}")
            cr_path = latest(OUTPUT_DIR / f"cfv2_retrieval_{tag}_{suffix}_*/predictions.csv")
            if cr_path is not None:
                c = pd.read_csv(cr_path)
                col = "pred_cfv2_retr_cal" if "pred_cfv2_retr_cal" in c.columns else (
                    "pred_cfv2_retr_raw" if "pred_cfv2_retr_raw" in c.columns else None)
                if col is not None:
                    c = c[["user_id", "movie_id", col]].rename(
                        columns={col: f"pred_cfv2_retr_{suffix}"})
                    df = df.merge(c, on=["user_id", "movie_id"], how="left")
                    df[f"pred_cfv2_retr_{suffix}"] = df[f"pred_cfv2_retr_{suffix}"].fillna(df["pred_cfv2_cal"]).clip(1.0, 5.0)
                    print(f"  cfv2_retr_{suffix}:      {cr_path}")

    # Metadata from original ratings
    ratings = load_ratings()
    train, _ = temporal_split(ratings)
    pop = item_popularity(train)
    item_mean = train.groupby("movie_id")["rating"].mean()
    user_mean_full = train.groupby("user_id")["rating"].mean()
    user_count = train.groupby("user_id").size()

    df["item_mean"] = df["movie_id"].map(item_mean).fillna(3.5)
    df["user_mean_hist"] = df["user_id"].map(user_mean_full).fillna(3.5)
    df["user_rating_count"] = df["user_id"].map(user_count).fillna(0).astype(int)
    df["log_pop"] = np.log1p(df["pop_count"])

    # Clip base-method preds to [1, 5]
    for c in ["pred_GlobalMean", "pred_UserMean", "pred_Baseline",
              "pred_SVD", "pred_ItemKNN", "pred_cfv2_cal", "pred_cfv2_raw"]:
        df[c] = df[c].clip(1.0, 5.0)

    # Derived features
    df["cfv2_delta_vs_knn"] = df["pred_cfv2_cal"] - df["pred_ItemKNN"]
    df["knn_vs_usermean"] = df["pred_ItemKNN"] - df["pred_UserMean"]
    df["pop_x_cfv2_delta"] = df["log_pop"] * df["cfv2_delta_vs_knn"]

    # LLM meta-features (signals from the CFv2 pipeline itself)
    if "chosen_pop_reliance" in df.columns:
        df["llm_pop_reliance"] = df["chosen_pop_reliance"].fillna(5.0).astype(float)
        df["llm_plausibility"] = df["chosen_plausibility"].fillna(5.0).astype(float)
        df["llm_tips_score"] = df["tips_pop_score"].fillna(0.0).astype(float)
        df["llm_delta_raw"] = df["delta"].fillna(0.0).astype(float)
        # Direction one-hot
        direction = df["direction"].fillna("same").astype(str).str.lower()
        df["dir_higher"] = (direction == "higher").astype(int)
        df["dir_lower"] = (direction == "lower").astype(int)
        # Confidence proxy: plausibility * (1 - pop_reliance/10) — intervention "quality"
        df["llm_confidence"] = df["llm_plausibility"] * (1.0 - df["llm_pop_reliance"] / 10.0)
    return df


FEATURE_SETS = {
    "cf_only": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
    ],
    "cf_plus_cfv2": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
    ],
    "cf_plus_cfv2_interact": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean", "pop_x_cfv2_delta",
    ],
    "all_plus_llm_meta": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean", "pop_x_cfv2_delta",
        "llm_pop_reliance", "llm_plausibility", "llm_tips_score",
        "llm_delta_raw", "dir_higher", "dir_lower", "llm_confidence",
    ],
    "all_plus_diverse": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal", "pred_diverse",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean",
    ],
    "everything": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal", "pred_diverse",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean", "pop_x_cfv2_delta",
        "llm_pop_reliance", "llm_plausibility", "llm_tips_score",
        "llm_delta_raw", "dir_higher", "dir_lower", "llm_confidence",
    ],
    "with_retrieval_and_neumf": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal", "pred_diverse",
        "pred_diverse_retrieval", "pred_cfv2_retr", "pred_NeuMF",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean",
    ],
    "max_all": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal", "pred_diverse",
        "pred_diverse_retrieval", "pred_cfv2_retr", "pred_NeuMF",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean", "pop_x_cfv2_delta",
        "llm_pop_reliance", "llm_plausibility", "llm_tips_score",
        "llm_delta_raw", "dir_higher", "dir_lower", "llm_confidence",
    ],
    "mega_both_models": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal", "pred_diverse",
        "pred_diverse_retr_14b", "pred_diverse_retr_72b",
        "pred_cfv2_retr_14b", "pred_cfv2_retr_72b",
        "pred_NeuMF",
        "item_mean", "user_mean_hist",
        "log_pop", "user_rating_count",
        "cfv2_delta_vs_knn", "knn_vs_usermean",
    ],
    # V2 feature sets
    "v2_cf_strong": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline", "pred_UserMean",
        "pred_NeuMF", "pred_LightGCN", "pred_LightGCN_llminit",
        "pred_BPRMF", "pred_TwoTower",
        "item_mean", "user_mean_hist", "log_pop", "user_rating_count",
    ],
    "v2_cf_plus_llm": [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline", "pred_UserMean",
        "pred_NeuMF", "pred_LightGCN", "pred_LightGCN_llminit",
        "pred_BPRMF", "pred_TwoTower",
        "pred_cfv2_cal", "pred_diverse", "pred_soft_dist",
        "pred_PreferenceCoT_cal", "pred_Analogical_cal",
        "item_mean", "user_mean_hist", "log_pop", "user_rating_count",
    ],
    "v2_all_everything": [
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
    ],
}


def _filter_existing(feats, df):
    """Drop feature names not present in df (e.g., retrieval/NeuMF not yet run)."""
    return [f for f in feats if f in df.columns]


def cv_score(X, y, model_factory, n_splits=5, seed=0):
    """5-fold CV. Return mean RMSE, MAE, bias of out-of-fold preds."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="n500",
                    help="matches cfv2_<tag>_* and cf_baseline_<tag>_* output dirs")
    ap.add_argument("--model_suffixes", type=str, default=None,
                    help="Comma-separated list of model suffixes to pull retrieval variants, e.g. '14b,72b'")
    args = ap.parse_args()
    model_suffixes = args.model_suffixes.split(",") if args.model_suffixes else None
    print(f"Loading predictions + metadata (tag={args.tag}, suffixes={model_suffixes})...")
    df = build_features(tag=args.tag, model_suffixes=model_suffixes)
    print(f"  n={len(df)}  mean_rating={df['rating'].mean():.3f}")
    y = df["rating"].astype(float).values

    # Baseline methods for comparison
    print("\n=== Base methods (no stacking) ===")
    base_methods = [
        "pred_ItemKNN", "pred_SVD", "pred_Baseline",
        "pred_UserMean", "pred_cfv2_cal",
    ]
    rows = []
    for m in base_methods:
        p = df[m].astype(float).values
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        mae = float(np.mean(np.abs(y - p)))
        bias = float(np.mean(p - y))
        rows.append({"method": m.replace("pred_", ""), "rmse": round(rmse, 4),
                     "mae": round(mae, 4), "bias": round(bias, 4)})
    print(pd.DataFrame(rows).sort_values("rmse").to_string(index=False))

    print("\n=== Stacked ridge regression (5-fold CV) ===")
    results = []
    for fs_name, feats in FEATURE_SETS.items():
        feats_avail = _filter_existing(feats, df)
        if len(feats_avail) < len(feats):
            missing = set(feats) - set(feats_avail)
            if not feats_avail:
                print(f"  skipping {fs_name}: all features missing")
                continue
            print(f"  {fs_name}: dropped missing features {missing}")
        feats = feats_avail
        X = df[feats].values.astype(float)
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            def _mk(a=alpha):
                return Ridge(alpha=a, random_state=0)
            rmse, mae, bias, _ = cv_score(X, y, _mk)
            results.append({"features": fs_name, "model": f"Ridge(α={alpha})",
                            "rmse": round(rmse, 4), "mae": round(mae, 4),
                            "bias": round(bias, 4), "n_feats": len(feats)})
        # RidgeCV auto-selects alpha
        def _mk_ridgecv():
            return RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3)
        rmse, mae, bias, _ = cv_score(X, y, _mk_ridgecv)
        results.append({"features": fs_name, "model": "RidgeCV",
                        "rmse": round(rmse, 4), "mae": round(mae, 4),
                        "bias": round(bias, 4), "n_feats": len(feats)})
    print(pd.DataFrame(results).sort_values("rmse").to_string(index=False))

    print("\n=== Gradient Boosting + Random Forest ===")
    nonlin_results = []
    for fs_name, feats in FEATURE_SETS.items():
        feats_avail = _filter_existing(feats, df)
        if not feats_avail:
            continue
        feats = feats_avail
        X = df[feats].values.astype(float)
        def _mk_gb():
            return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                             random_state=0)
        rmse, mae, bias, _ = cv_score(X, y, _mk_gb)
        nonlin_results.append({"features": fs_name, "model": "GradientBoost",
                               "rmse": round(rmse, 4), "mae": round(mae, 4),
                               "bias": round(bias, 4), "n_feats": len(feats)})
        def _mk_rf():
            return RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                         random_state=0, n_jobs=-1)
        rmse, mae, bias, _ = cv_score(X, y, _mk_rf)
        nonlin_results.append({"features": fs_name, "model": "RandomForest",
                               "rmse": round(rmse, 4), "mae": round(mae, 4),
                               "bias": round(bias, 4), "n_feats": len(feats)})
    print(pd.DataFrame(nonlin_results).sort_values("rmse").to_string(index=False))

    # Best stacker — refit on full data, report feature weights (only for linear models)
    print("\n=== Best stacker: coefficient inspection (RidgeCV on richest feature set) ===")
    for candidate_key in ("max_all", "with_retrieval_and_neumf", "everything", "cf_plus_cfv2_interact"):
        candidate = _filter_existing(FEATURE_SETS[candidate_key], df)
        if len(candidate) >= 5:
            best_feats = candidate
            print(f"  using feature set: {candidate_key} ({len(best_feats)} feats)")
            break
    X = df[best_feats].values.astype(float)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=3)
    ridge.fit(X, y)
    coef_tbl = pd.DataFrame({
        "feature": best_feats,
        "coef": np.round(ridge.coef_, 4),
    }).sort_values("coef", key=abs, ascending=False)
    print(f"  Selected alpha: {ridge.alpha_:.4f}   intercept: {ridge.intercept_:.4f}")
    print(coef_tbl.to_string(index=False))

    out_dir = Path(OUTPUT_DIR) / "stacker"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(results + nonlin_results).to_csv(out_dir / "cv_scores.csv", index=False)
    coef_tbl.to_csv(out_dir / "ridge_coefs.csv", index=False)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
