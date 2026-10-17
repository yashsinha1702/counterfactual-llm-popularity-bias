"""Re-analyze already-saved predictions on items with very low training
popularity (< 20 ratings, < 10 ratings) to find the true cold-start
threshold beyond which LLMs could compete with ItemKNN."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR


def load_merged():
    fp = OUTPUT_DIR / "cold_start_cold_start" / "merged_predictions.csv"
    df = pd.read_csv(fp)
    return df


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    df = load_merged()
    print(f"Merged pairs: {len(df)}")

    methods = ["ItemKNN", "SVD", "UserMean", "diverse_v2_14B",
               "diverse_72B", "hybrid_KNN_diverse_05"]

    thresholds = [5, 10, 20, 40, 80, 200]
    print("\n=== RMSE by pop_count ≤ threshold (truly cold regime) ===")
    rows = []
    for thr in thresholds:
        sub = df[df["pop_count"] <= thr]
        if len(sub) < 5:
            continue
        y = sub["rating"].astype(float).values
        row = {"threshold_≤": thr, "n_pairs": len(sub),
               "mean_pop": float(sub["pop_count"].mean())}
        for m in methods:
            if m in sub.columns:
                p = np.clip(sub[m].astype(float).values, 1.0, 5.0)
                row[m] = round(rmse(y, p), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    print("\n=== Same but pop_count RANGE slices (ring) ===")
    ranges = [(0, 10), (10, 30), (30, 80), (80, 200), (200, 500), (500, 10000)]
    rows = []
    for lo, hi in ranges:
        sub = df[(df["pop_count"] > lo) & (df["pop_count"] <= hi)]
        if len(sub) < 5:
            rows.append({"range": f"({lo}, {hi}]", "n": len(sub)})
            continue
        y = sub["rating"].astype(float).values
        row = {"range": f"({lo}, {hi}]", "n": len(sub),
               "mean_pop": float(sub["pop_count"].mean())}
        for m in methods:
            if m in sub.columns:
                p = np.clip(sub[m].astype(float).values, 1.0, 5.0)
                row[m] = round(rmse(y, p), 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
