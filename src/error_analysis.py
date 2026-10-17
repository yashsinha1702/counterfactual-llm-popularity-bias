"""Qualitative error analysis of baseline vs counterfactual predictions.

Reads the predictions.csv from a run and surfaces:
  - pairs where CF moved the prediction most
  - how often the move was correctional vs. harmful
  - the chosen explanation's direction and popularity-reliance distribution
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(df):
    df = df.copy()
    df["err_b"] = df["pred_baseline"] - df["rating"]
    df["err_cf"] = df["pred_cf"] - df["rating"]
    df["abs_err_b"] = df["err_b"].abs()
    df["abs_err_cf"] = df["err_cf"].abs()
    df["delta"] = df["pred_cf"] - df["pred_baseline"]
    df["improved"] = df["abs_err_cf"] < df["abs_err_b"]
    df["unchanged"] = np.isclose(df["pred_cf"], df["pred_baseline"])
    return df


def print_top_deltas(df, n=20):
    top = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(n)
    cols = ["user_id", "movie_id", "pop_count", "rating",
            "pred_baseline", "pred_cf",
            "cf_chosen_popreliance", "cf_chosen_plausibility", "cf_tips_pop_score"]
    print("Top deltas (|pred_cf - pred_baseline|):")
    print(top[cols].to_string(index=False))
    return top


def aggregate(df):
    changed = df[~df["unchanged"]]
    improved = changed[changed["improved"]]
    harmed = changed[~changed["improved"]]
    print("\n=== Aggregate: CF vs baseline ===")
    print(f"  total pairs:          {len(df)}")
    print(f"  unchanged:            {len(df) - len(changed)} ({(len(df) - len(changed)) / len(df):.0%})")
    print(f"  changed:              {len(changed)} ({len(changed) / len(df):.0%})")
    print(f"    improved (closer):  {len(improved)} ({len(improved) / max(1, len(changed)):.0%} of changed)")
    print(f"    harmed  (farther):  {len(harmed)} ({len(harmed) / max(1, len(changed)):.0%} of changed)")
    print(f"  mean |err| baseline:  {df['abs_err_b'].mean():.3f}")
    print(f"  mean |err| CF:        {df['abs_err_cf'].mean():.3f}")
    print(f"  CF moved prediction toward truth by {(df['abs_err_b'].mean() - df['abs_err_cf'].mean()):+.3f} MAE on average")


def bucket_improvements(df, n_buckets=5):
    q = pd.qcut(np.log1p(df["pop_count"]), q=n_buckets, labels=False, duplicates="drop")
    df2 = df.assign(_b=q)
    print("\n=== Improvement by popularity bucket ===")
    rows = []
    for b in sorted(df2["_b"].unique()):
        m = df2["_b"] == b
        sub = df2[m]
        changed = sub[~sub["unchanged"]]
        improved = changed[changed["improved"]]
        rows.append({
            "bucket": int(b),
            "n": len(sub),
            "mean_pop": sub["pop_count"].mean(),
            "pct_changed": f"{len(changed) / max(1, len(sub)):.0%}",
            "pct_improved_of_changed": f"{len(improved) / max(1, len(changed)):.0%}" if len(changed) else "-",
            "mean_abs_err_baseline": sub["abs_err_b"].mean(),
            "mean_abs_err_cf": sub["abs_err_cf"].mean(),
            "bias_baseline": sub["err_b"].mean(),
            "bias_cf": sub["err_cf"].mean(),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def print_examples(df, n_each=3):
    df = df[~df["unchanged"]].copy()
    if df.empty:
        print("No changed pairs.")
        return
    print(f"\n=== {n_each} examples where CF helped (biggest improvement) ===")
    helped = df[df["improved"]].assign(gain=df["abs_err_b"] - df["abs_err_cf"])
    helped = helped.sort_values("gain", ascending=False).head(n_each)
    for _, r in helped.iterrows():
        print(f"\n  user={int(r['user_id'])} movie={int(r['movie_id'])} pop={int(r['pop_count'])} "
              f"true={r['rating']} | baseline={r['pred_baseline']} -> cf={r['pred_cf']} "
              f"(|err|: {r['abs_err_b']:.1f} -> {r['abs_err_cf']:.1f})")
        print(f"    baseline reason: {str(r.get('baseline_reason',''))[:220]}")
        print(f"    cf chosen (plaus={r['cf_chosen_plausibility']:.0f}, pop_rel={r['cf_chosen_popreliance']:.0f}): "
              f"{str(r.get('cf_chosen_text',''))[:220]}")

    print(f"\n=== {n_each} examples where CF hurt (biggest harm) ===")
    hurt = df[~df["improved"]].assign(loss=df["abs_err_cf"] - df["abs_err_b"])
    hurt = hurt.sort_values("loss", ascending=False).head(n_each)
    for _, r in hurt.iterrows():
        print(f"\n  user={int(r['user_id'])} movie={int(r['movie_id'])} pop={int(r['pop_count'])} "
              f"true={r['rating']} | baseline={r['pred_baseline']} -> cf={r['pred_cf']} "
              f"(|err|: {r['abs_err_b']:.1f} -> {r['abs_err_cf']:.1f})")
        print(f"    baseline reason: {str(r.get('baseline_reason',''))[:220]}")
        print(f"    cf chosen (plaus={r['cf_chosen_plausibility']:.0f}, pop_rel={r['cf_chosen_popreliance']:.0f}): "
              f"{str(r.get('cf_chosen_text',''))[:220]}")


def pop_reliance_dist(df):
    print("\n=== Distribution of chosen hypothesis popularity-reliance scores ===")
    print(df["cf_chosen_popreliance"].describe())
    print("\nMean plausibility of chosen:", df["cf_chosen_plausibility"].mean())
    print("Mean TIPS pop_score of target item:", df["cf_tips_pop_score"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True)
    ap.add_argument("--n_top", type=int, default=15)
    ap.add_argument("--n_examples", type=int, default=3)
    args = ap.parse_args()
    df = pd.read_csv(args.predictions)
    df = summarize(df)
    aggregate(df)
    bucket_improvements(df)
    pop_reliance_dist(df)
    print_top_deltas(df, n=args.n_top)
    print_examples(df, n_each=args.n_examples)


if __name__ == "__main__":
    main()
