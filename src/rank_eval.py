"""Compute HR@k and NDCG@k for every method, from cached per-pair predictions.

Evaluation protocol:
- Binarise rating: >=4 -> positive, <4 -> negative.
- For each user with >=1 positive and >=2 total items in the n=2000 sample,
  rank items by each method's predicted score (descending).
- HR@k = 1 if any positive is in the top-k ranking, else 0.
- NDCG@k = DCG@k / IDCG@k where relevance = 1 if positive else 0.
- Averaged across users.

We also report per-popularity-bucket breakdowns to see whether the counterfactual
framework helps niche items rank higher.

Usage:
    python3 -u src/rank_eval.py
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
POS_THR = 4            # rating >= 4 means positive
KS = [1, 2, 3, 5, 10, 20]       # top-k values to report — small k needed since users have ~5-7 items

# ---------------------------------------------------------------
# 1. Load all predictions into one master table
# ---------------------------------------------------------------
# For each method we want the single "canonical" prediction column,
# joined on (user_id, movie_id, rating, timestamp, pop_count).

METHOD_FILES = [
    # (folder, [columns_to_keep])
    ('cf_baseline_n2000_1776428101',
     ['pred_GlobalMean', 'pred_UserMean', 'pred_Baseline', 'pred_SVD', 'pred_ItemKNN']),
    ('cfv2_n2000_1776428102',
     ['pred_cfv2_raw', 'pred_cfv2_cal']),
    ('cfv2_retrieval_n2000_14b_1776446882',
     ['pred_cfv2_retr_raw', 'pred_cfv2_retr_cal']),
    ('cfv2_retrieval_n2000_72b_1776450114',
     ['pred_cfv2_retr_raw', 'pred_cfv2_retr_cal']),
    ('diverse_n2000_1776442083',          ['pred_diverse']),
    ('diverse_retrieval_n2000_14b_1776446353', ['pred_diverse_retrieval']),
    ('diverse_retrieval_n2000_72b_1776448595', ['pred_diverse_retrieval']),
    ('soft_dist_n2000_1776555345',        ['pred_soft_dist']),
    ('preference_cot_n2000_1776556256',
     ['pred_PreferenceCoT_raw', 'pred_PreferenceCoT_cal']),
    ('confidence_router_n2000_1776558381', ['pred_ConfidenceRouter']),
    ('neumf_n2000_1776531265',            ['pred_NeuMF']),     # tuned dim=64
    ('neumf_ens_v4_n2000_1776745616',     ['pred_NeuMF_ens_avg']),
    ('bprmf_n2000_1776535451',            ['pred_BPRMF']),
    ('lightgcn_n2000_1776539447',         ['pred_LightGCN']),
    ('lightgcn_llminit_n2000_1776669346', ['pred_LightGCN_llminit']),
    ('twotower_n2000_1776552445',         ['pred_TwoTower']),
    ('deepfm_n2000_1776709933',           ['pred_DeepFM_avg']),
]

# Some method names collide across files (e.g. pred_cfv2_retr_cal in both 14b and 72b folders).
# Rename them to be unambiguous.
RENAME = {
    ('cfv2_retrieval_n2000_14b_1776446882', 'pred_cfv2_retr_raw'): 'pred_cfv2_retr_raw_14b',
    ('cfv2_retrieval_n2000_14b_1776446882', 'pred_cfv2_retr_cal'): 'pred_cfv2_retr_cal_14b',
    ('cfv2_retrieval_n2000_72b_1776450114', 'pred_cfv2_retr_raw'): 'pred_cfv2_retr_raw_72b',
    ('cfv2_retrieval_n2000_72b_1776450114', 'pred_cfv2_retr_cal'): 'pred_cfv2_retr_cal_72b',
    ('diverse_retrieval_n2000_14b_1776446353', 'pred_diverse_retrieval'): 'pred_diverse_retr_14b',
    ('diverse_retrieval_n2000_72b_1776448595', 'pred_diverse_retrieval'): 'pred_diverse_retr_72b',
}

t0 = time.time()
print(f'[{time.time()-t0:6.1f}s] Loading master prediction table...', flush=True)

master = None
for folder, cols in METHOD_FILES:
    p = os.path.join(OUT, folder, 'predictions.csv')
    if not os.path.exists(p):
        print(f'  [skip] {folder} — file missing', flush=True)
        continue
    df = pd.read_csv(p)
    keep = ['user_id', 'movie_id', 'rating', 'timestamp', 'pop_count'] + cols
    df = df[[c for c in keep if c in df.columns]].copy()
    # Apply renames
    df = df.rename(columns={old: new for (fol, old), new in RENAME.items() if fol == folder})
    if master is None:
        master = df
    else:
        merge_keys = ['user_id', 'movie_id']
        pred_only = [c for c in df.columns if c.startswith('pred_')]
        master = master.merge(df[merge_keys + pred_only], on=merge_keys, how='outer')
    print(f'  [loaded] {folder}: +{len(df)} rows, +{len([c for c in df.columns if c.startswith("pred_")])} pred cols', flush=True)

print(f'[{time.time()-t0:6.1f}s] master shape: {master.shape}', flush=True)
print(f'[{time.time()-t0:6.1f}s] rating coverage: {master["rating"].notna().sum()}/{len(master)}', flush=True)

# Drop rows missing the ground-truth rating (shouldn't happen if outer-merge coverage is right)
master = master.dropna(subset=['rating']).reset_index(drop=True)
master['positive'] = (master['rating'] >= POS_THR).astype(int)

pred_cols = [c for c in master.columns if c.startswith('pred_')]
print(f'[{time.time()-t0:6.1f}s] {len(pred_cols)} prediction columns, '
      f'{master["positive"].sum()} positives / {len(master)} total '
      f'({master["positive"].mean():.1%})', flush=True)


# ---------------------------------------------------------------
# 2. Per-user ranking HR/NDCG
# ---------------------------------------------------------------
def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(rels_ranked, k):
    """rels_ranked: relevance (0/1) of items in the predicted ranking order."""
    rels_ranked = np.asarray(rels_ranked)[:k]
    if rels_ranked.sum() == 0:
        return 0.0
    ideal = np.sort(rels_ranked)[::-1]
    ideal_dcg = dcg(ideal[:k])
    if ideal_dcg == 0:
        return 0.0
    return dcg(rels_ranked[:k]) / ideal_dcg


def hr_at_k(rels_ranked, k):
    rels_ranked = np.asarray(rels_ranked)[:k]
    return 1.0 if rels_ranked.sum() > 0 else 0.0


def recall_at_k(rels_ranked, k, total_pos):
    rels_ranked = np.asarray(rels_ranked)[:k]
    return float(rels_ranked.sum() / total_pos) if total_pos else 0.0


def mrr(rels_ranked):
    """Mean Reciprocal Rank — 1/rank of first positive."""
    rels_ranked = np.asarray(rels_ranked)
    pos_idx = np.where(rels_ranked == 1)[0]
    if len(pos_idx) == 0:
        return 0.0
    return 1.0 / (pos_idx[0] + 1)


# Pre-compute popularity buckets for per-bucket breakdowns
master['pop_bucket'] = pd.qcut(master['pop_count'], q=5, labels=False, duplicates='drop')


# ---------------------------------------------------------------
# 3. Evaluate each method
# ---------------------------------------------------------------
print(f'\n[{time.time()-t0:6.1f}s] Evaluating ranking metrics per method...\n', flush=True)

# Group once by user — index list per user
user_groups = master.groupby('user_id').groups

# Build a user index table
eligible_users = []
for uid, idx in user_groups.items():
    grp = master.loc[idx]
    n_pos = int(grp['positive'].sum())
    n_total = len(grp)
    if n_pos >= 1 and n_total >= 2:
        eligible_users.append(uid)

print(f'[{time.time()-t0:6.1f}s] eligible users: {len(eligible_users)} '
      f'(>=1 positive & >=2 total items)\n', flush=True)

# For per-bucket breakdown we compute HR@10 for positives in each bucket
results = []
per_bucket = []

for method in pred_cols:
    overall = {f'hr@{k}': [] for k in KS}
    overall.update({f'ndcg@{k}': [] for k in KS})
    overall.update({f'recall@{k}': [] for k in KS})
    overall['mrr'] = []

    # per-bucket: HR@10 and NDCG@10 by bucket of positives
    bucket_hr = defaultdict(list)
    bucket_ndcg = defaultdict(list)

    skipped = 0
    for uid in eligible_users:
        idx = user_groups[uid]
        grp = master.loc[idx]
        preds = grp[method].values
        if np.isnan(preds).any():
            skipped += 1
            continue
        rels = grp['positive'].values
        buckets = grp['pop_bucket'].values

        order = np.argsort(-preds, kind='stable')
        rels_ranked = rels[order]
        buckets_ranked = buckets[order]

        n_pos = int(rels.sum())

        for k in KS:
            overall[f'hr@{k}'].append(hr_at_k(rels_ranked, k))
            overall[f'ndcg@{k}'].append(ndcg_at_k(rels_ranked, k))
            overall[f'recall@{k}'].append(recall_at_k(rels_ranked, k, n_pos))
        overall['mrr'].append(mrr(rels_ranked))

        # per-bucket HR@10 / NDCG@10 — one entry per positive, assigned to its bucket
        for rank_pos, (rel, b) in enumerate(zip(rels_ranked, buckets_ranked)):
            if rel == 1 and not np.isnan(b):
                # did this positive make top-10?
                b = int(b)
                bucket_hr[b].append(1.0 if rank_pos < 10 else 0.0)
                # contribution to NDCG@10 if positive is at rank_pos < 10
                disc = 1.0 / np.log2(rank_pos + 2) if rank_pos < 10 else 0.0
                bucket_ndcg[b].append(disc)

    row = {'method': method, 'users_evaluated': len(eligible_users) - skipped, 'users_skipped': skipped}
    for metric, vals in overall.items():
        row[metric] = float(np.mean(vals)) if vals else float('nan')
    results.append(row)

    for b in range(5):
        if bucket_hr[b]:
            per_bucket.append({
                'method': method, 'bucket': b,
                'n_positives': len(bucket_hr[b]),
                'hr@10': float(np.mean(bucket_hr[b])),
                'ndcg@10_contrib': float(np.mean(bucket_ndcg[b])),
            })

    print(f'[{time.time()-t0:6.1f}s] {method:<35} '
          f'HR@1={row["hr@1"]:.4f}  HR@3={row["hr@3"]:.4f}  '
          f'NDCG@5={row["ndcg@5"]:.4f}  NDCG@10={row["ndcg@10"]:.4f}  '
          f'MRR={row["mrr"]:.4f}  '
          f'(skipped {skipped} users)', flush=True)


# ---------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------
out_dir = os.path.join(OUT, 'ranking_eval')
os.makedirs(out_dir, exist_ok=True)

results_df = pd.DataFrame(results).sort_values('ndcg@10', ascending=False)
results_df.to_csv(os.path.join(out_dir, 'ranking_results.csv'), index=False)

per_bucket_df = pd.DataFrame(per_bucket)
per_bucket_df.to_csv(os.path.join(out_dir, 'ranking_per_bucket.csv'), index=False)

# Save a summary json with top line
with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    summary = {
        'protocol': 'rank-within-sample, n=2000 stratified test sample',
        'positive_threshold': POS_THR,
        'eligible_users': len(eligible_users),
        'n_pairs': int(len(master)),
        'positives_per_user_median': float(
            master.groupby('user_id')['positive'].sum().loc[eligible_users].median()
        ),
        'n_methods': len(results),
        'top_3_by_ndcg10': results_df.head(3)[['method', 'ndcg@10', 'hr@10']].to_dict('records'),
    }
    json.dump(summary, f, indent=2)

print(f'\n[{time.time()-t0:6.1f}s] Done. Wrote:', flush=True)
print(f'  {out_dir}/ranking_results.csv', flush=True)
print(f'  {out_dir}/ranking_per_bucket.csv', flush=True)
print(f'  {out_dir}/summary.json', flush=True)

print('\n=== Top 10 by NDCG@10 ===', flush=True)
cols = ['method', 'hr@1', 'hr@3', 'hr@5', 'ndcg@5', 'ndcg@10', 'mrr']
print(results_df[cols].head(10).to_string(index=False), flush=True)
print('\n=== Bottom 5 by NDCG@10 ===', flush=True)
print(results_df[cols].tail(5).to_string(index=False), flush=True)
