"""Merge predictions from every run and report per-popularity-bucket
RMSE / MAE / bias for all methods side-by-side, to answer:

  Does any LLM method beat classical CF on the niche/cold regime?

No new LLM calls — operates entirely on saved predictions.csv files.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, POPULARITY_BUCKETS


def latest_run(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def load_predictions():
    runs = {
        "main_72b": latest_run(str(OUTPUT_DIR / "n500_*/predictions.csv")),
        "diverse_72b": latest_run(str(OUTPUT_DIR / "diverse_n500_*/predictions.csv")),
        "cf_baseline": latest_run(str(OUTPUT_DIR / "cf_baseline_n500_*/predictions.csv")),
        "improved_14b": latest_run(str(OUTPUT_DIR / "improved_n500_*/predictions.csv")),
        "suite_14b_main": latest_run(str(OUTPUT_DIR / "14b_n500_*/predictions.csv")),
    }
    dfs = {}
    for name, path in runs.items():
        if path is None:
            print(f"  MISSING: {name}")
            continue
        print(f"  loaded {name}: {path}")
        dfs[name] = pd.read_csv(path)
    return dfs


def merge_all(dfs):
    def key(df):
        return df[["user_id", "movie_id"]].astype(int)

    base = None
    keep_renames = {
        "main_72b": [("pred_baseline", "baseline_72B"),
                     ("pred_cf", "CF_pipeline_72B"),
                     ("rating", "rating"), ("pop_count", "pop_count")],
        "diverse_72b": [("pred_diverse", "diverse_72B")],
        "cf_baseline": [("pred_GlobalMean", "GlobalMean"),
                        ("pred_UserMean", "UserMean"),
                        ("pred_Baseline", "Baseline_CF"),
                        ("pred_SVD", "SVD"),
                        ("pred_ItemKNN", "ItemKNN")],
        "improved_14b": [("pred_diverse_v2", "diverse_v2_14B"),
                         ("pred_logprob_no_stats", "logprob_14B"),
                         ("pred_logprob_stats", "logprob_stats_14B"),
                         ("pred_fixed_SVD_diverse_v2_a0.5", "hybrid_SVD_diverse_05"),
                         ("pred_fixed_ItemKNN_diverse_v2_a0.5", "hybrid_KNN_diverse_05")],
        "suite_14b_main": [("pred_baseline", "baseline_14B"),
                           ("pred_cf", "CF_pipeline_14B")],
    }
    for name, mapping in keep_renames.items():
        if name not in dfs:
            continue
        df = dfs[name].copy()
        df["user_id"] = df["user_id"].astype(int)
        df["movie_id"] = df["movie_id"].astype(int)
        cols = ["user_id", "movie_id"] + [c for c, _ in mapping if c in df.columns]
        sub = df[cols].rename(columns={c: alias for c, alias in mapping if c in df.columns})
        if base is None:
            base = sub
        else:
            common = [c for c in sub.columns if c in base.columns and c not in ("user_id", "movie_id")]
            sub = sub.drop(columns=common)
            base = base.merge(sub, on=["user_id", "movie_id"], how="inner")
    return base


def bucket_metrics(df, methods, n_buckets=POPULARITY_BUCKETS):
    y = df["rating"].astype(float).values
    pop = df["pop_count"].astype(float).values
    try:
        q = pd.qcut(np.log1p(pop), q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        q = np.zeros_like(pop, dtype=int)
    q = pd.Series(q).fillna(0).astype(int).values
    rows = []
    for m in methods:
        if m not in df.columns:
            continue
        p = np.clip(df[m].astype(float).values, 1.0, 5.0)
        for b in sorted(np.unique(q)):
            mask = q == b
            rows.append({
                "method": m, "bucket": int(b), "n": int(mask.sum()),
                "mean_pop": float(pop[mask].mean()),
                "rmse": float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2))),
                "mae": float(np.mean(np.abs(y[mask] - p[mask]))),
                "bias": float(np.mean(p[mask] - y[mask])),
            })
    return pd.DataFrame(rows)


def overall_metrics(df, methods):
    y = df["rating"].astype(float).values
    rows = []
    for m in methods:
        if m not in df.columns:
            continue
        p = np.clip(df[m].astype(float).values, 1.0, 5.0)
        rows.append({
            "method": m, "n": len(df),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "mae": float(np.mean(np.abs(y - p))),
            "bias": float(np.mean(p - y)),
        })
    return pd.DataFrame(rows).sort_values("rmse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_tag", default="cold_start")
    args = ap.parse_args()

    print("Loading predictions...")
    dfs = load_predictions()
    df = merge_all(dfs)
    print(f"\nMerged: {len(df)} pairs across {len(df.columns)-4} methods")

    methods = [
        "GlobalMean", "UserMean", "Baseline_CF", "SVD", "ItemKNN",
        "baseline_72B", "CF_pipeline_72B", "diverse_72B",
        "baseline_14B", "CF_pipeline_14B", "diverse_v2_14B",
        "logprob_14B", "logprob_stats_14B",
        "hybrid_SVD_diverse_05", "hybrid_KNN_diverse_05",
    ]

    overall = overall_metrics(df, methods)
    print("\n=== Overall RMSE/MAE/bias (merged sample, all methods) ===")
    print(overall.to_string(index=False))

    buckets = bucket_metrics(df, methods)
    pivot_rmse = buckets.pivot(index="method", columns="bucket", values="rmse").round(3)
    pivot_rmse.columns = [f"b{int(c)} (rmse)" for c in pivot_rmse.columns]
    pivot_bias = buckets.pivot(index="method", columns="bucket", values="bias").round(3)
    pivot_bias.columns = [f"b{int(c)} (bias)" for c in pivot_bias.columns]

    pivot_rmse = pivot_rmse.reindex([m for m in methods if m in pivot_rmse.index])
    pivot_bias = pivot_bias.reindex([m for m in methods if m in pivot_bias.index])

    print("\n=== RMSE by popularity bucket (b0 = niche ... b4 = blockbuster) ===")
    print(pivot_rmse.to_string())

    print("\n=== Bias by popularity bucket ===")
    print(pivot_bias.to_string())

    # Find winner per bucket
    print("\n=== Winner per bucket (by RMSE) ===")
    for b in sorted(buckets["bucket"].unique()):
        sub = buckets[buckets["bucket"] == b].sort_values("rmse")
        top3 = sub.head(3)
        mean_pop = float(sub.iloc[0]["mean_pop"])
        print(f"  bucket_{int(b)} (n≈{int(sub.iloc[0]['n'])}, mean_pop={mean_pop:.0f}):")
        for _, row in top3.iterrows():
            print(f"      {row['method']:25s}  rmse={row['rmse']:.3f}  mae={row['mae']:.3f}  bias={row['bias']:+.3f}")

    out_dir = Path(OUTPUT_DIR) / f"cold_start_{args.save_tag}"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "merged_predictions.csv", index=False)
    overall.to_csv(out_dir / "overall.csv", index=False)
    buckets.to_csv(out_dir / "by_bucket.csv", index=False)
    pivot_rmse.to_csv(out_dir / "rmse_by_bucket.csv")
    pivot_bias.to_csv(out_dir / "bias_by_bucket.csv")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
