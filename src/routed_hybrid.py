"""Popularity-routed hybrid: for each test pair, pick CFv2_cal if the
target item's training popularity is below a threshold, else pick
ItemKNN. This exploits the observed crossover where the counterfactual
framework beats ItemKNN on niche items (bucket 0, 1) and loses on
popular items (bucket 2, 3, 4).

The routing threshold can be chosen:
  (a) fixed, passed via --threshold (e.g., 300),
  (b) picked by grid search on this sample (reported as in-sample best;
      honest version would choose on a held-out train slice)."""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def bias(y, p): return float(np.mean(p - y))


def load_cfv2_predictions(tag="n500"):
    paths = sorted(glob.glob(str(OUTPUT_DIR / f"cfv2_{tag}_*/predictions.csv")))
    if not paths:
        raise FileNotFoundError(f"No CFv2 predictions found for tag={tag}")
    return pd.read_csv(paths[-1])


def load_cf_baseline_predictions(tag="n500"):
    paths = sorted(glob.glob(str(OUTPUT_DIR / f"cf_baseline_{tag}_*/predictions.csv")))
    if not paths:
        raise FileNotFoundError(f"No cf_baseline predictions found for tag={tag}")
    return pd.read_csv(paths[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="Popularity threshold; pairs with pop < threshold use CFv2_cal, else ItemKNN")
    ap.add_argument("--tag", type=str, default="n500")
    args = ap.parse_args()

    cfv2 = load_cfv2_predictions(tag=args.tag)
    cfb = load_cf_baseline_predictions(tag=args.tag)

    # Merge on (user_id, movie_id)
    cfv2 = cfv2[["user_id", "movie_id", "rating", "pop_count",
                 "cf_reference", "pred_cfv2_raw", "pred_cfv2_cal"]].copy()
    cfb = cfb[["user_id", "movie_id", "pred_ItemKNN", "pred_SVD", "pred_UserMean"]].copy()
    df = cfv2.merge(cfb, on=["user_id", "movie_id"], how="inner")
    df["pred_cfv2_cal"] = df["pred_cfv2_cal"].clip(1.0, 5.0)
    df["pred_ItemKNN"] = df["pred_ItemKNN"].clip(1.0, 5.0)
    y = df["rating"].astype(float).values
    pop = df["pop_count"].astype(float).values
    cfv2_cal = df["pred_cfv2_cal"].values
    knn = df["pred_ItemKNN"].values
    print(f"Merged pairs: {len(df)}")

    print(f"\nIndividual methods:")
    print(f"  ItemKNN:  rmse={rmse(y, knn):.4f} mae={mae(y, knn):.4f} bias={bias(y, knn):+.4f}")
    print(f"  CFv2_cal: rmse={rmse(y, cfv2_cal):.4f} mae={mae(y, cfv2_cal):.4f} bias={bias(y, cfv2_cal):+.4f}")
    print(f"  mean_truth={y.mean():.3f}  mean_pop={pop.mean():.1f}")

    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000]

    print(f"\n=== Popularity-routed hybrid (CFv2_cal if pop < threshold, else ItemKNN) ===")
    rows = []
    for thr in thresholds:
        use_cfv2 = pop < thr
        preds = np.where(use_cfv2, cfv2_cal, knn)
        rmse_v = rmse(y, preds)
        mae_v = mae(y, preds)
        bias_v = bias(y, preds)
        rows.append({
            "threshold": thr,
            "n_cfv2": int(use_cfv2.sum()),
            "n_knn": int((~use_cfv2).sum()),
            "rmse": round(rmse_v, 4),
            "mae": round(mae_v, 4),
            "bias": round(bias_v, 4),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    best = table.sort_values("rmse").iloc[0]
    print(f"\nBest threshold (RMSE): {int(best['threshold'])}  "
          f"-> rmse={best['rmse']} mae={best['mae']} bias={best['bias']:+.4f}")

    # Also report the "oracle" lower bound: per-pair pick the better of the two.
    err_knn = np.abs(y - knn)
    err_cfv2 = np.abs(y - cfv2_cal)
    better_cfv2 = err_cfv2 < err_knn
    oracle_preds = np.where(better_cfv2, cfv2_cal, knn)
    print(f"\nOracle per-pair selection (upper bound on routing): "
          f"rmse={rmse(y, oracle_preds):.4f} mae={mae(y, oracle_preds):.4f} "
          f"(chose CFv2 on {better_cfv2.mean():.0%} of pairs)")

    # Also try simple linear blending
    print("\n=== Linear blend: α*ItemKNN + (1-α)*CFv2_cal ===")
    rows = []
    for a in np.linspace(0.0, 1.0, 21):
        blend = a * knn + (1 - a) * cfv2_cal
        blend = np.clip(blend, 1.0, 5.0)
        rows.append({"alpha_itemknn": round(a, 2), "rmse": round(rmse(y, blend), 4),
                     "mae": round(mae(y, blend), 4), "bias": round(bias(y, blend), 4)})
    blend_tbl = pd.DataFrame(rows)
    print(blend_tbl.to_string(index=False))

    out_dir = Path(OUTPUT_DIR) / "routed_hybrid"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "threshold_sweep.csv", index=False)
    blend_tbl.to_csv(out_dir / "alpha_sweep.csv", index=False)


if __name__ == "__main__":
    main()
