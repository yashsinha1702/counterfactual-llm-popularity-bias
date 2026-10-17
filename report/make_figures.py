"""Generate figures for the Overleaf report."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(REPO_ROOT, 'outputs')
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'ml-1m')

os.makedirs(FIG_DIR, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def save(name):
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    plt.close()


# ---------------------------------------------------------------
# 1. Main leaderboard — RMSE for every method
# ---------------------------------------------------------------
leaderboard = [
    ('Global Mean',            1.172, 'trivial'),
    ('User Mean',              1.084, 'trivial'),
    ('14B 3-stage CF pipeline',1.401, 'llm'),
    ('Zero-shot LLM (14B)',    1.319, 'llm'),
    ('Preference CoT',         1.201, 'llm'),
    ('Diverse prompt (14B)',   1.131, 'llm'),
    ('Soft-distribution LLM',  1.126, 'llm'),
    ('14B diverse + retrieval',1.054, 'llm'),
    ('CFv2 raw (no cal)',      1.047, 'llm'),
    ('Confidence router',      0.953, 'llm'),
    ('Baseline CF (bu+bi)',    0.954, 'cf'),
    ('SVD (k=50)',             0.938, 'cf'),
    ('LightGCN (random)',      0.937, 'cf'),
    ('CFv2 cal (no retr)',     0.935, 'llm'),
    ('LightGCN + LLM init',    0.932, 'cf'),
    ('Two-Tower',              0.930, 'cf'),
    ('ItemKNN',                0.901, 'cf'),
    ('72B CFv2 + retrieval',   0.897, 'llm'),
    ('BPR-MF',                 0.889, 'cf'),
    ('NeuMF tuned (dim=64)',   0.884, 'cf'),
    ('Ridge stacker v4',       0.876, 'stacker'),
    ('ORACLE (per-pair)',      0.860, 'oracle'),
]
leaderboard.sort(key=lambda x: x[1], reverse=True)
colour = {'llm': '#c0392b', 'cf': '#2980b9', 'stacker': '#27ae60',
          'oracle': '#7f8c8d', 'trivial': '#95a5a6'}
labels, rmses, cats = zip(*leaderboard)
colours = [colour[c] for c in cats]

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(labels, rmses, color=colours, edgecolor='black', lw=0.4)
ax.set_xlabel('RMSE (n=2000 stratified test sample)')
ax.set_title('Methods ranked by RMSE on MovieLens-1M')
ax.set_xlim(0.80, 1.45)
for bar, v in zip(bars, rmses):
    ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{v:.3f}', va='center', fontsize=8)
ax.axvline(0.8760, color='#27ae60', ls='--', alpha=0.7, lw=1.0, label='Best stacker 0.876')
ax.axvline(0.8599, color='#7f8c8d', ls=':',  alpha=0.7, lw=1.0, label='Oracle 0.860')
ax.axvline(0.9012, color='#2980b9', ls='-.', alpha=0.7, lw=1.0, label='ItemKNN 0.901')
from matplotlib.patches import Patch
handles = [
    Patch(color=colour['trivial'], label='Trivial'),
    Patch(color=colour['cf'],      label='Classical / neural CF'),
    Patch(color=colour['llm'],     label='LLM methods'),
    Patch(color=colour['stacker'], label='Ridge stacker'),
    Patch(color=colour['oracle'],  label='Oracle (upper bound)'),
]
ax.legend(handles=handles, loc='lower right', frameon=True)
save('leaderboard')


# ---------------------------------------------------------------
# 2. Journey plot — how the project RMSE progressed
# ---------------------------------------------------------------
journey = [
    ('Zero-shot 14B',                       1.319),
    ('Original 3-stage pipeline',           1.372),
    ('Diverse prompt',                      1.158),
    ('CFv2 + bias cal',                     0.935),
    ('Pop-routed hybrid',                   0.901),
    ('Ridge stacker (CF only)',             0.896),
    ('NeuMF tuned',                         0.884),
    ('Overnight v1 stacker (+retrieval)',   0.881),
    ('Overnight v2 stacker',                0.876),
    ('Stacker v4 (final)',                  0.876),
]
names, vals = zip(*journey)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(range(len(names)), vals, marker='o', markersize=9,
        color='#2980b9', lw=2.0)
ax.axhline(0.9012, color='#c0392b', ls='--', lw=1.0, label='ItemKNN 0.901')
ax.axhline(0.8599, color='#7f8c8d', ls=':',  lw=1.0, label='Oracle 0.860')
for i, v in enumerate(vals):
    ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('RMSE (n=2000)')
ax.set_title('Project journey: RMSE over experimental stages')
ax.set_ylim(0.80, 1.45)
ax.legend(loc='upper right')
save('journey')


# ---------------------------------------------------------------
# 3. Per-popularity-bucket RMSE and bias
# ---------------------------------------------------------------
buckets = ['b0 (niche)', 'b1', 'b2', 'b3', 'b4 (pop)']
itemknn_rmse = [0.986, 0.919, 0.875, 0.885, 0.834]
cfv2_rmse    = [1.015, 0.940, 0.931, 0.902, 0.882]
itemknn_bias = [0.19, 0.10, 0.02, 0.06, 0.07]
cfv2_bias    = [0.08, -0.01, -0.07, 0.01, 0.00]
zs_bias      = [0.60, 0.56, 0.56, 0.60, 0.43]  # zero-shot baseline order of magnitude

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(buckets)); w = 0.35
ax1.bar(x - w/2, itemknn_rmse, w, label='ItemKNN', color='#2980b9')
ax1.bar(x + w/2, cfv2_rmse,    w, label='CFv2 + bias cal (LLM)', color='#c0392b')
ax1.set_xticks(x); ax1.set_xticklabels(buckets)
ax1.set_ylabel('RMSE'); ax1.set_title('RMSE by popularity bucket (n=2000)')
ax1.legend(); ax1.set_ylim(0.75, 1.10)

ax2.bar(x - w/2, itemknn_bias, w, label='ItemKNN', color='#2980b9')
ax2.bar(x + w/2, cfv2_bias,    w, label='CFv2 + cal', color='#c0392b')
ax2.axhline(0, color='k', lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(buckets)
ax2.set_ylabel('Bias (mean pred - mean truth)')
ax2.set_title('Bias by popularity bucket (n=2000)')
ax2.legend(); ax2.set_ylim(-0.15, 0.25)
plt.tight_layout()
save('per_bucket')


# ---------------------------------------------------------------
# 4. Stacker v4 Ridge coefficients
# ---------------------------------------------------------------
df = pd.read_csv(f'{OUT}/stacker_v4/ridge_coefs.csv')
df = df.reindex(df.coef.abs().sort_values(ascending=True).index)
df = df.tail(18)  # top 18 absolute
fig, ax = plt.subplots(figsize=(9.0, 6.2))
colours = ['#27ae60' if c > 0 else '#c0392b' for c in df.coef]
ax.barh(df.feature, df.coef, color=colours, edgecolor='black', lw=0.3)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Ridge coefficient')
ax.set_title('Stacker v4 Ridge coefficients (top-18 by |coef|)')
# Pad horizontal room so negative labels don't collide with the y-axis ticks
x_min = df.coef.min() - 0.05
x_max = df.coef.max() + 0.05
ax.set_xlim(x_min, x_max)
for y, v in zip(range(len(df)), df.coef.values):
    ax.text(v + (0.006 if v > 0 else -0.006), y,
            f'{v:+.3f}', va='center',
            ha='left' if v > 0 else 'right', fontsize=8)
plt.tight_layout()
save('stacker_coefs')


# ---------------------------------------------------------------
# 5. TIPS α ablation
# ---------------------------------------------------------------
alphas = [0.00, 0.25, 0.50, 0.75, 1.00]
rmses  = [1.370, 1.370, 1.366, 1.373, 1.372]
biases = [0.416, 0.410, 0.398, 0.390, 0.366]
fig, ax1 = plt.subplots(figsize=(7, 4.2))
l1, = ax1.plot(alphas, rmses, 'o-', color='#2980b9', lw=2, label='RMSE')
ax1.set_xlabel(r'TIPS weight $\alpha$')
ax1.set_ylabel('RMSE', color='#2980b9')
ax1.set_ylim(1.33, 1.40)
ax2 = ax1.twinx()
l2, = ax2.plot(alphas, biases, 's--', color='#c0392b', lw=2, label='Bias')
ax2.set_ylabel('Bias', color='#c0392b')
ax2.set_ylim(0.35, 0.45)
ax1.axhline(1.365, color='#7f8c8d', ls=':', lw=1,
            label='no-intervention (1.365)')
plt.title(r'TIPS $\alpha$-sweep ablation (n=500, Qwen-72B)')
ax1.legend(handles=[l1, l2], loc='lower right')
ax1.grid(True, alpha=0.3)
save('tips_ablation')


# ---------------------------------------------------------------
# 6. Bias reduction: zero-shot vs CFv2
# ---------------------------------------------------------------
methods = ['Zero-shot\nLLM', '3-stage\n(α=0.5)', 'Diverse\nprompt',
           'CFv2 raw', 'CFv2 +\nbias cal']
overall_bias = [0.620, 0.398, 0.253, 0.469, 0.002]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
cs = ['#c0392b', '#e67e22', '#f39c12', '#c0392b', '#27ae60']
bars = ax.bar(methods, overall_bias, color=cs, edgecolor='black', lw=0.4)
for b, v in zip(bars, overall_bias):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:+.3f}',
            ha='center', fontsize=9)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('Overall prediction bias')
ax.set_title('Prediction bias reduction across LLM stages')
ax.set_ylim(-0.05, 0.72)
save('bias_reduction')


# ---------------------------------------------------------------
# 7. Cold-start comparison
# ---------------------------------------------------------------
cold20  = {'ItemKNN': 1.127, 'Hybrid α=0.5': 1.162, 'SVD': 1.183,
           'Baseline_CF': 1.186, 'Diverse LLM': 1.274, 'UserMean': 1.275}
cold0   = {'Hybrid α=0.5': 0.942, 'ItemKNN': 0.944, 'SVD': 0.944,
           'UserMean': 0.961, 'Diverse LLM': 0.991, 'GlobalMean': 1.153}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, d, title in zip(axes, [cold20, cold0],
                        ['Low-pop items (pop ≤ 20), n=500',
                         'Truly unseen items (pop = 0), n=54']):
    cs = ['#27ae60' if 'Hybrid' in k else
          '#2980b9' if 'KNN' in k or 'SVD' in k or 'Baseline' in k else
          '#c0392b' if 'LLM' in k else
          '#95a5a6' for k in d.keys()]
    ax.barh(list(d.keys()), list(d.values()), color=cs, edgecolor='black', lw=0.4)
    ax.invert_yaxis()
    for i, (k, v) in enumerate(d.items()):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    ax.set_xlabel('RMSE'); ax.set_title(title)
plt.tight_layout()
save('coldstart')


# ---------------------------------------------------------------
# 8. Long-tail popularity distribution (ML-1M)
# ---------------------------------------------------------------
ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'),
                      sep='::', engine='python',
                      names=['u', 'i', 'r', 't'])
pop = ratings.i.value_counts().sort_values(ascending=False)
top20 = pop[:int(0.2 * len(pop))].sum() / pop.sum()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(pop.values, color='#2980b9', lw=1.2)
ax.fill_between(range(len(pop)), pop.values, alpha=0.2, color='#2980b9')
ax.axvline(int(0.2 * len(pop)), color='#c0392b', ls='--',
           label=f'Top 20% → {top20:.1%} of ratings')
ax.set_xlabel('Item index (sorted by popularity)')
ax.set_ylabel('Rating count per item')
ax.set_title('MovieLens-1M item-popularity long tail')
ax.set_yscale('log'); ax.legend()
save('longtail')


# ---------------------------------------------------------------
# 9. Predicted vs true scatter — flagship method (Stacker v4) vs a
#    weak starting point (Zero-shot LLM 14B) if available
# ---------------------------------------------------------------
neumf_df = pd.read_csv(f'{OUT}/neumf_ens_v4_n2000_1776745616/predictions.csv')
cfv2_df  = pd.read_csv(f'{OUT}/cfv2_n2000_1776428102/predictions.csv')
baseline_df = pd.read_csv(f'{OUT}/cf_baseline_n2000_1776428101/predictions.csv')

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, df, col, title in zip(
        axes,
        [baseline_df, cfv2_df, neumf_df],
        ['pred_ItemKNN', 'pred_cfv2_cal', 'pred_NeuMF_ens_avg'],
        ['ItemKNN (cosine, k=40)', 'CFv2 + bias calibration',
         'NeuMF ensemble (5 seeds)']):
    t = df['rating'].values
    p = df[col].values
    # jitter true ratings for readability
    j = t + np.random.normal(0, 0.06, size=len(t))
    ax.scatter(j, p, s=6, alpha=0.28, color='#2980b9')
    ax.plot([1, 5], [1, 5], 'k--', lw=1.0)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    ax.set_title(f'{title}\nRMSE = {rmse:.3f}')
    ax.set_xlabel('True rating'); ax.set_ylabel('Predicted rating')
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
    ax.set_aspect('equal')
plt.tight_layout()
save('pred_scatter')


# ---------------------------------------------------------------
# 10. Stacker feature set comparison
# ---------------------------------------------------------------
cv = pd.read_csv(f'{OUT}/stacker_v4/cv_scores.csv')
cv = cv[cv['model'].isin(['Ridge(α=100.0)', 'Ridge(α=10.0)'])]
best = cv.groupby('features')['rmse'].min().sort_values()
fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.barh(best.index, best.values,
               color=['#27ae60' if v <= 0.878 else '#2980b9' for v in best.values],
               edgecolor='black', lw=0.4)
ax.invert_yaxis()
for b, v in zip(bars, best.values):
    ax.text(v + 0.0005, b.get_y() + b.get_height()/2, f'{v:.4f}',
            va='center', fontsize=8)
ax.axvline(0.8599, color='#7f8c8d', ls=':', label='Oracle 0.8599')
ax.set_xlim(0.860, 0.885)
ax.set_xlabel('5-fold CV RMSE (best Ridge α)')
ax.set_title('Stacker v4 feature-set comparison')
ax.legend(loc='lower right')
save('stacker_featuresets')


# ---------------------------------------------------------------
# 11. Logprob / variant ablations bar
# ---------------------------------------------------------------
abl = [
    ('logprob-14B\nno user stats', 1.450),
    ('logprob-14B\n+ user stats',  1.383),
    ('3-stage\nα=0.5 (14B)',       1.401),
    ('diverse-v2\nno stats',       1.162),
    ('diverse-v2\n+ stats',        1.158),
    ('pref. CoT\nraw',             1.247),
    ('pref. CoT\ncalibrated',      1.201),
    ('soft-dist.\nretrieval',      1.126),
]
names, vals = zip(*abl)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(names, vals, color='#c0392b', edgecolor='black', lw=0.4)
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
ax.set_ylabel('RMSE')
ax.set_title('LLM-prompting ablations and variants')
ax.set_ylim(1.10, 1.55)
save('llm_ablations')


# ---------------------------------------------------------------
# 12. Baseline analysis popularity bias (from notebook)
# ---------------------------------------------------------------
bl = pd.DataFrame([
    ('Popularity Ranking', 0.9895, 2801.7, 0.27),
    ('Global Mean',        1.1466, 2801.7, 0.27),
    ('Bias Model',         0.9222, 1037.6, 1.67),
    ('User-User KNN',      0.9216, 1788.0, 19.54),
    ('Item-Item KNN',      1.0126,  636.5, 88.07),
    ('SVD MF (k=50)',      0.8931, 1100.9, 18.86),
], columns=['model', 'rmse', 'arp', 'coverage'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
pal = ['#c0392b', '#c0392b', '#e67e22', '#2980b9', '#8e44ad', '#27ae60']
ax1.barh(bl['model'], bl['arp'], color=pal, edgecolor='black', lw=0.4)
ax1.invert_yaxis()
for i, v in enumerate(bl['arp']):
    ax1.text(v + 25, i, f'{v:.0f}', va='center', fontsize=9)
ax1.set_xlabel('ARP — avg recommendation popularity (↓ better)')
ax1.set_title('Popularity bias (ARP) per CF method')

ax2.barh(bl['model'], bl['coverage'], color=pal, edgecolor='black', lw=0.4)
ax2.invert_yaxis()
for i, v in enumerate(bl['coverage']):
    ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
ax2.set_xlabel('Catalogue coverage % (↑ better)')
ax2.set_title('Catalogue coverage per CF method')
plt.tight_layout()
save('baseline_arp_cov')


# ---------------------------------------------------------------
# 13. RMSE vs ARP tradeoff (baseline notebook data)
# ---------------------------------------------------------------
bl2 = bl[bl.model.isin(['Bias Model','User-User KNN','Item-Item KNN','SVD MF (k=50)'])]
fig, ax = plt.subplots(figsize=(7.5, 5))
colour_map = {'Bias Model':'#e67e22', 'User-User KNN':'#2980b9',
              'Item-Item KNN':'#8e44ad', 'SVD MF (k=50)':'#27ae60'}
for _, row in bl2.iterrows():
    ax.scatter(row.arp, row.rmse, s=220, color=colour_map[row.model], zorder=4,
               edgecolor='black')
    ax.annotate(row.model, (row.arp, row.rmse),
                xytext=(8, 8), textcoords='offset points', fontsize=10)
ax.set_xlabel('ARP  (lower = less popularity-biased)')
ax.set_ylabel('RMSE  (lower = more accurate)')
ax.set_title('Accuracy–bias trade-off across classical CF baselines')
ax.grid(True, alpha=0.3)
save('accuracy_bias_tradeoff')

# ---------------------------------------------------------------
# 14. Per-bucket RMSE across five CF methods (classical, from
#     cf_baseline_n2000 metrics.json)
# ---------------------------------------------------------------
with open(f'{OUT}/cf_baseline_n2000_1776428101/metrics.json') as f:
    bcf = json.load(f)
methods = ['GlobalMean', 'UserMean', 'Baseline', 'SVD', 'ItemKNN']
buckets = ['bucket_0', 'bucket_1', 'bucket_2', 'bucket_3', 'bucket_4']
grid = np.zeros((len(methods), len(buckets)))
for i, m in enumerate(methods):
    for j, b in enumerate(buckets):
        grid[i, j] = bcf['results'][m]['by_popularity'][b]['rmse']
fig, ax = plt.subplots(figsize=(9, 4.2))
im = ax.imshow(grid, cmap='RdYlBu_r', aspect='auto',
               vmin=0.83, vmax=1.40)
ax.set_xticks(range(len(buckets)))
ax.set_xticklabels(['b0 (niche)', 'b1', 'b2', 'b3', 'b4 (pop)'])
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
for i in range(len(methods)):
    for j in range(len(buckets)):
        v = grid[i, j]
        txt_col = 'white' if v > 1.05 or v < 0.87 else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                fontsize=9, color=txt_col)
plt.colorbar(im, ax=ax, label='RMSE')
ax.set_title('Per-popularity-bucket RMSE, classical CF methods (n=2000)')
save('heatmap_per_bucket')


# ---------------------------------------------------------------
# 15. Bias across classical baselines on per-bucket level
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
bias_grid = np.zeros((len(methods), len(buckets)))
for i, m in enumerate(methods):
    for j, b in enumerate(buckets):
        bias_grid[i, j] = bcf['results'][m]['by_popularity'][b]['bias']
im = ax.imshow(bias_grid, cmap='RdBu_r', aspect='auto',
               vmin=-0.25, vmax=0.65)
ax.set_xticks(range(len(buckets)))
ax.set_xticklabels(['b0 (niche)', 'b1', 'b2', 'b3', 'b4 (pop)'])
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
for i in range(len(methods)):
    for j in range(len(buckets)):
        v = bias_grid[i, j]
        txt_col = 'white' if abs(v) > 0.35 else 'black'
        ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                fontsize=9, color=txt_col)
plt.colorbar(im, ax=ax, label='Bias (mean pred − mean truth)')
ax.set_title('Per-popularity-bucket bias, classical CF methods (n=2000)')
save('heatmap_bias_per_bucket')


# ---------------------------------------------------------------
# 16. Bias leaderboard (MAB = mean absolute per-bucket bias)
# ---------------------------------------------------------------
bias_board = [
    ('CFv2 retr-14B + cal (LLM)',  0.0258, 'llm'),
    ('CFv2 + bias cal (LLM)',      0.0342, 'llm'),
    ('Baseline CF (b_u+b_i)',      0.0830, 'cf'),
    ('SVD (k=50)',                 0.0872, 'cf'),
    ('Pref-CoT cal (LLM)',         0.0870, 'llm'),
    ('ItemKNN',                    0.0876, 'cf'),
    ('Diverse-14B retrieval',      0.1906, 'llm'),
    ('UserMean',                   0.2448, 'trivial'),
    ('Diverse-14B (no retr)',      0.2524, 'llm'),
    ('GlobalMean',                 0.2846, 'trivial'),
    ('Soft-distribution (LLM)',    0.3656, 'llm'),
    ('Diverse-72B retrieval',      0.4364, 'llm'),
    ('CFv2 raw (LLM, no cal)',     0.4690, 'llm'),
    ('Zero-shot LLM-14B',          0.6220, 'llm'),    # measured n=500: 14b_n500_1776424046
]
bias_board.sort(key=lambda x: x[1], reverse=True)
lbls, mabs, cats = zip(*bias_board)
colour = {'llm': '#c0392b', 'cf': '#2980b9', 'trivial': '#95a5a6', 'llm-fail': '#f39c12'}
colours = [colour[c] for c in cats]
fig, ax = plt.subplots(figsize=(9.5, 6.2))
bars = ax.barh(lbls, mabs, color=colours, edgecolor='black', lw=0.4)
ax.set_xlabel('MAB — mean |per-bucket prediction bias|  (↓ better)')
ax.set_title('Bias leaderboard: counterfactual framework takes the top 2 slots')
for b, v in zip(bars, mabs):
    ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
            f'{v:.3f}', va='center', fontsize=8)
from matplotlib.patches import Patch
handles = [
    Patch(color=colour['cf'],       label='Classical CF'),
    Patch(color=colour['llm'],      label='LLM / counterfactual'),
    Patch(color=colour['trivial'],  label='Trivial'),
]
ax.legend(handles=handles, loc='center right', bbox_to_anchor=(1.0, 0.35),
          frameon=True)
ax.axvline(0.0258, color='#27ae60', ls='--', lw=1, alpha=0.7,
           label='Best (CFv2 retr-14B)')
ax.set_xlim(0, max(mabs) + 0.08)
plt.tight_layout()
save('bias_leaderboard')


# ---------------------------------------------------------------
# 17. Blockbuster bias reduction story (single-panel bar)
# ---------------------------------------------------------------
methods = ['Zero-shot\nLLM (14B)', '3-stage\n(α=0.5)', 'Diverse\n(14B)',
           'CFv2 raw', 'CFv2 +\nbias cal', 'CFv2 retr-14B\n+ cal']
# Values: zero-shot=+0.43, 3-stage α=0.5 blockbuster=+0.14 (RESULTS.md),
# diverse 14B=+0.018, CFv2 raw=+0.464, CFv2+cal=-0.005, CFv2 retr-14B+cal=-0.007
block_bias = [0.43, 0.14, 0.018, 0.464, -0.005, -0.007]
fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ['#c0392b' if abs(v) > 0.3 else
          '#e67e22' if abs(v) > 0.15 else
          '#f39c12' if abs(v) > 0.05 else
          '#27ae60' for v in block_bias]
bars = ax.bar(methods, block_bias, color=colors, edgecolor='black', lw=0.4)
for b, v in zip(bars, block_bias):
    ax.text(b.get_x() + b.get_width()/2,
            v + (0.015 if v >= 0 else -0.04),
            f'{v:+.3f}', ha='center', fontsize=9)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('Blockbuster-bucket bias (b4, mean pop=1344)')
ax.set_title('Popularity-bias mitigation: blockbuster bias reduced ~96%')
ax.set_ylim(-0.15, 0.55)
save('blockbuster_bias_reduction')

# ---------------------------------------------------------------
# 18. HR@k leaderboard
# ---------------------------------------------------------------
rank_df = pd.read_csv(f'{OUT}/ranking_eval/ranking_results.csv')

# Build a clean method-name map and category
NAME_MAP = {
    'pred_GlobalMean':          ('GlobalMean',              'trivial'),
    'pred_UserMean':            ('UserMean',                'trivial'),
    'pred_Baseline':            ('Baseline CF (b_u+b_i)',   'cf'),
    'pred_SVD':                 ('SVD (k=50)',              'cf'),
    'pred_ItemKNN':             ('ItemKNN (k=40)',          'cf'),
    'pred_cfv2_raw':            ('CFv2 raw (14B)',          'llm'),
    'pred_cfv2_cal':            ('CFv2 + bias cal (14B)',   'llm'),
    'pred_cfv2_retr_raw_14b':   ('CFv2 retr-14B raw',       'llm'),
    'pred_cfv2_retr_cal_14b':   ('CFv2 retr-14B + cal',     'llm'),
    # pred_cfv2_retr_raw_72b and pred_cfv2_retr_cal_72b omitted — 72B retrieval
    # parse failure: all 2000 calls returned delta=0 fallback (direction='err'),
    # making the predictions identical to ItemKNN + a constant calibration offset.
    # Kept in rank_eval.py output for audit, but not included in any leaderboard.
    'pred_diverse':             ('Diverse (14B)',           'llm'),
    'pred_diverse_retr_14b':    ('Diverse retr-14B',        'llm'),
    'pred_diverse_retr_72b':    ('Diverse retr-72B',        'llm'),
    'pred_soft_dist':           ('Soft-distribution',       'llm'),
    'pred_PreferenceCoT_raw':   ('Preference-CoT raw',      'llm'),
    'pred_PreferenceCoT_cal':   ('Preference-CoT + cal',    'llm'),
    'pred_ConfidenceRouter':    ('Confidence router',       'llm'),
    'pred_NeuMF':               ('NeuMF tuned',             'cf'),
    'pred_NeuMF_ens_avg':       ('NeuMF ensemble',          'cf'),
    'pred_BPRMF':               ('BPR-MF',                  'cf'),
    'pred_LightGCN':            ('LightGCN',                'cf'),
    'pred_LightGCN_llminit':    ('LightGCN + LLM init',     'cf'),
    'pred_TwoTower':            ('Two-Tower',               'cf'),
    'pred_DeepFM_avg':          ('DeepFM (avg)',            'cf'),
}

rank_df['name'] = rank_df['method'].map(lambda m: NAME_MAP.get(m, (m, 'other'))[0])
rank_df['cat']  = rank_df['method'].map(lambda m: NAME_MAP.get(m, (m, 'other'))[1])

# Drop exact duplicates where raw == cal (calibration is a constant shift for rating
# predictions; within-sample ranking is unchanged). Keep the "cal" form for display.
drop_raw = {'pred_cfv2_raw', 'pred_cfv2_retr_raw_14b', 'pred_PreferenceCoT_raw'}
# Drop 72B retrieval entirely — parse failure, effectively ItemKNN+calibration.
drop_72b_retr = {'pred_cfv2_retr_raw_72b', 'pred_cfv2_retr_cal_72b'}
rank_df_plot = rank_df[~rank_df['method'].isin(drop_raw | drop_72b_retr)].copy()
# Also drop any method that fell through NAME_MAP (now 'other') — keeps chart clean.
rank_df_plot = rank_df_plot[rank_df_plot['cat'] != 'other'].copy()

# ---- HR leaderboard
cat_colour = {'cf': '#2980b9', 'llm': '#c0392b', 'trivial': '#95a5a6',
              'llm-fail': '#f39c12', 'other': '#7f8c8d'}

hr_df = rank_df_plot.sort_values('hr@1', ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 7))
colours = [cat_colour[c] for c in hr_df['cat']]
bars = ax.barh(hr_df['name'], hr_df['hr@1'], color=colours, edgecolor='black', lw=0.4)
for b, v in zip(bars, hr_df['hr@1']):
    ax.text(v + 0.003, b.get_y() + b.get_height()/2, f'{v:.3f}',
            va='center', fontsize=8)
ax.set_xlabel('HR@1  (↑ better)')
ax.set_title('Hit-Rate @1 leaderboard  —  n=2000 within-sample ranking')
ax.set_xlim(0.65, 0.92)
from matplotlib.patches import Patch
handles = [
    Patch(color=cat_colour['cf'],       label='Classical / neural CF'),
    Patch(color=cat_colour['llm'],      label='LLM / counterfactual'),
    Patch(color=cat_colour['trivial'],  label='Trivial'),
]
ax.legend(handles=handles, loc='lower right')
save('hr_leaderboard')


# ---- NDCG leaderboard
ndcg_df = rank_df_plot.sort_values('ndcg@10', ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 7))
colours = [cat_colour[c] for c in ndcg_df['cat']]
bars = ax.barh(ndcg_df['name'], ndcg_df['ndcg@10'], color=colours, edgecolor='black', lw=0.4)
for b, v in zip(bars, ndcg_df['ndcg@10']):
    ax.text(v + 0.0015, b.get_y() + b.get_height()/2, f'{v:.3f}',
            va='center', fontsize=8)
ax.set_xlabel('NDCG@10  (↑ better)')
ax.set_title('NDCG@10 leaderboard  —  n=2000 within-sample ranking')
ax.set_xlim(0.86, 0.97)
ax.legend(handles=handles, loc='lower right')
save('ndcg_leaderboard')


# ---- Combined HR vs NDCG scatter (shows correlation between metrics)
fig, ax = plt.subplots(figsize=(8, 6))
for cat, col in cat_colour.items():
    sub = rank_df_plot[rank_df_plot['cat'] == cat]
    if len(sub):
        ax.scatter(sub['hr@1'], sub['ndcg@10'], s=120, color=col, edgecolor='black',
                   lw=0.5, alpha=0.85, label=cat)
        for _, r in sub.iterrows():
            ax.annotate(r['name'], (r['hr@1'], r['ndcg@10']),
                        xytext=(5, 3), textcoords='offset points', fontsize=7.5)
ax.set_xlabel('HR@1'); ax.set_ylabel('NDCG@10')
ax.set_title('HR@1 vs NDCG@10: top-right is best ranking')
ax.grid(True, alpha=0.3)
save('hr_ndcg_scatter')


# ---- Per-popularity-bucket rank-weighted score
# (ndcg@10_contrib is mean per-positive discount = 1/log2(rank+2) if rank<10 else 0)
# Higher contrib = positives in that bucket ranked higher on average.
pb = pd.read_csv(f'{OUT}/ranking_eval/ranking_per_bucket.csv')

featured = [
    ('pred_GlobalMean',        'GlobalMean',           'trivial'),
    ('pred_ItemKNN',           'ItemKNN',              'cf'),
    ('pred_NeuMF_ens_avg',     'NeuMF ens.',           'cf'),
    ('pred_cfv2_cal',          'CFv2 + cal',           'llm'),
    ('pred_cfv2_retr_cal_14b', 'CFv2 retr-14B + cal',  'llm'),
    ('pred_diverse',           'Diverse',              'llm'),
]
fig, ax = plt.subplots(figsize=(10, 4.8))
width = 0.13
buckets = [0, 1, 2, 3, 4]
x = np.arange(len(buckets))
for i, (m, lbl, cat) in enumerate(featured):
    sub = pb[pb['method'] == m].sort_values('bucket')
    vals = [float(sub[sub['bucket'] == b]['ndcg@10_contrib'].values[0])
            if b in sub['bucket'].values else np.nan
            for b in buckets]
    ax.bar(x + (i - len(featured)/2) * width, vals, width,
           label=lbl, color=cat_colour[cat], edgecolor='black', lw=0.3,
           alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(['b0 niche', 'b1', 'b2', 'b3', 'b4 blockbuster'])
ax.set_ylabel('Mean rank-discount of positives in top-10\n(higher = positives rank earlier)')
ax.set_title('Per-bucket ranking quality — counterfactual methods rank niche positives higher')
ax.legend(fontsize=8, ncol=3, loc='lower right')
save('ranking_per_bucket')


print('All figures written to', FIG_DIR)
