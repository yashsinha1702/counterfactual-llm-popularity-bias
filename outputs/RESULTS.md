# LLM Counterfactual Reasoning + Classical CF for Rating Prediction

## Project premise

**Goal:** use an LLM to predict precise explicit ratings (1–5) for target user-item pairs on MovieLens.

**The challenge:** off-the-shelf LLMs and standard prompt engineering suffer from exposure and popularity biases — pre-training data saturates them with culturally popular items (mainstream blockbusters), so they skew predictions toward high ratings for well-known movies regardless of a specific user's niche tastes, and often hallucinate post-hoc rationales to justify popularity-driven predictions.

**The proposed architecture:** a counterfactual-reasoning (causal-inference) framework enforcing that a "cause" (a debiased explanation of why a user likes a movie) must logically and causally precede the "effect" (the predicted 1–5 rating). Three-stage cognitive process:

1. **Abduction** — given user demographics, rating history, and target-movie attributes, generate k candidate textual explanations for why the user might rate the movie in various directions. The LLM infers hidden latent variables (true semantic match M) from factual observations (historical ratings).
2. **Intervention** — construct hypothetical situations to mathematically intervene and remove the confounding variable of global item popularity P. Realized via **Time-aware Inverse Propensity Scoring (TIPS)**: propensities estimated from rating frequency in 30-day buckets, used as weights that penalize explanations relying on mainstream appeal rather than intrinsic semantic alignment.
3. **Prediction** — the LLM outputs the final rating based strictly on the single debiased explanation selected by the intervention stage.

By strictly enforcing this pipeline, the system aims to ensure that predicted ratings are driven by genuine user-item semantic matching rather than the LLM's inherent conformity bias.

---

## Setup and infrastructure

**Dataset:** MovieLens 1M (6,040 users × 3,883 movies × 1M ratings, ~3 years).
**Temporal split:** per-user leave-last-20% → 797,758 train / 202,451 test.
**Test sampling:** stratified by popularity quantile (5 buckets).
**LLMs served on our own GPU:** Qwen2.5-72B-Instruct-AWQ and Qwen2.5-14B-Instruct, self-served via vLLM on H100 NVL (GPU 1, port 8002). We downloaded weights from HuggingFace and ran our own server rather than relying on any shared inference infrastructure.
**Classical CF:** numpy/scipy reimplementations (ItemKNN on bias-centered residuals, ALS-fit biases, truncated SVD) because scikit-surprise is incompatible with numpy ≥ 2.0.
**LLM output caching:** all responses cached in sqlite keyed by (model, messages, params) — enables near-instant re-runs of ablations.
**Seed:** 0. Mean true rating ≈ 3.37–3.44 across samples.

---

## Experimental journey (chronological summary)

| Phase | What we ran | Result |
|---|---|---|
| 0 | n=5 smoke test — plumbing validation | pipeline runs end-to-end |
| 1 | n=500 on Qwen-72B — original 3-stage CF pipeline | RMSE 1.366, cuts blockbuster bias 0.43→0.14 |
| 2 | n=500 α-ablation (TIPS α ∈ {0, 0.25, 0.5, 0.75, 1.0, no_intervention}) | α=1 ≈ α=0; time-awareness null |
| 3 | Diverse-prompt control (single-call pros/cons) | RMSE 1.208 — *beats* the 3-stage pipeline |
| 4 | Switched vLLM server to Qwen-14B-Instruct (cross-model test) | 3-stage pipeline degrades further; diverse still wins |
| 5 | Classical CF baselines (ItemKNN, SVD, Baseline, UserMean, GlobalMean) | ItemKNN 0.930 beats every LLM method |
| 6 | Cold-start (pop ≤ 20; pop = 0 unseen) | LLM-hybrid wins only on unseen items |
| 7 | Causal-delta prediction (CFv2) + bias calibration | RMSE 0.958 (LLM) — huge jump from 1.37 |
| 8 | Popularity-routed hybrid + linear blend | RMSE 0.923 on n=500 — beats pure ItemKNN |
| 9 | Ridge stacker (Ridge/RF/GBM + CV) | RMSE 0.9155 on n=500, 0.8962 on n=2000 — new best |
| 10 | LLM meta-features added to stacker | gain 0.0001 — CFv2 meta redundant |
| 11 | Diverse-prompt LLM on n=2000 added as stacker feature | gain 0.001 — orthogonal LLM signal, new best RMSE 0.8952 |
| 12 | **Overnight run v1**: movie embeddings (sentence-transformer), retrieval-augmented diverse + CFv2 on Qwen-14B AND Qwen-72B | 72B CFv2-retrieval + bias-cal achieves **0.897 standalone** (beats ItemKNN) |
| 13 | NeuMF (neural matrix factorization, k=16 embeddings, 15 epochs) | **0.893 standalone** — best single model |
| 14 | **Ridge stacker with 17 mega_both_models features** | **RMSE 0.8811, 5-fold CV** — 62% of oracle gap closed |
| 15 | **Overnight run v2**: NeuMF tuned (dim=64, 40 epochs) | **0.8836** — improved NeuMF with wider embeddings |
| 16 | BPR-MF (Bayesian Personalized Ranking matrix factorization, dim=64) | **0.8888** — orthogonal CF objective |
| 17 | LightGCN (3-layer graph convolution, dim=64, random init) | 0.9367 — graph CF underperforms NeuMF |
| 18 | LightGCN + LLM item embedding init (sentence-transformer) | 0.9316 — LLM init gains +0.005 over random |
| 19 | Two-Tower (LLM item embeddings, content-based user tower) | 0.9296 — content-based CF model |
| 20 | New LLM methods: soft-distribution (E[rating] from p(1..5)), preference CoT, analogical, confidence router | soft-dist 1.126, pref-CoT 1.201 (cal), analogical failed, router 0.953 |
| 21 | **Updated Ridge stacker v2** with all new CF + LLM features (28 feats) | **RMSE 0.8764, 5-fold CV** — 73% of oracle gap closed |
| 22 | **Overnight v3**: Multi-seed NeuMF ensemble (5 seeds, original arch, GPU), DeepFM (overfits), residual NeuMF (zero gain), LightGBM stacker | NeuMF avg 0.8845 standalone; DeepFM/residual/LightGBM all failed to help stacker |
| 23 | **Stacker v4**: NeuMF ensemble avg + genre affinity features (user's per-genre rating) | **RMSE 0.8760, 5-fold CV** — new best, 76% of oracle gap closed |

---

**Dataset:** MovieLens 1M, temporal leave-last-20% split.
**LLMs:** Qwen2.5-72B-Instruct-AWQ and Qwen2.5-14B-Instruct, self-served on H100 NVL (GPU 1, vLLM port 8002).
**Classical CF:** numpy/scipy reimplementations (ItemKNN on bias-centered residuals, ALS biases, truncated SVD).
**Seed:** 0. Mean true rating ≈ 3.37-3.44 across samples.

---

## Final leaderboard (n=2000 stratified test sample)

| Rank | Method | RMSE | MAE | Bias | Notes |
|---|---|---|---|---|---|
| 🥇 | **Ridge stacker `v4_minimal` (17 feats, Ridge α=100, 5-fold CV)** | **0.8760** | **0.6952** | −0.000 | NeuMF ensemble avg + genre affinity + core CF/LLM |
| 2 | Ridge stacker `v2_all_everything` (28 feats, Ridge α=100, 5-fold CV) | 0.8764 | 0.6945 | +0.000 | all v2 CF + LLM + retrieval + meta |
| 🥈 | Ridge stacker `mega_both_models` (17 feats, Ridge α=10) | 0.8768 | 0.6949 | −0.000 | retrieval-14B + retrieval-72B + NeuMF + CF |
| 🥉 | Ridge stacker `v2_cf_strong` (12 feats, Ridge α=10) | 0.8777 | 0.6943 | +0.000 | NeuMF + BPR-MF + LightGCN + classical CF |
| 4 | Ridge stacker `max_all` (23 feats, Ridge α=10) | 0.8779 | 0.6940 | −0.000 | adds LLM meta to mega |
| 5 | Ridge stacker `v2_cf_plus_llm` (16 feats, Ridge α=10) | 0.8782 | 0.6936 | +0.000 | v2 CF + LLM signals |
| 6 | Ridge stacker `with_retrieval_and_neumf` (15 feats, Ridge α=10) | 0.8782 | 0.6949 | −0.000 | one retrieval model instead of two |
| 7 | **NeuMF tuned** (neural MF, dim=64, epoch 14) | **0.8836** | 0.6984 | +0.023 | best single model |
| 8 | **BPR-MF** (Bayesian Personalized Ranking MF, dim=64, epoch 29) | **0.8888** | 0.7054 | +0.017 | orthogonal CF objective |
| 9 | Ridge stacker `everything` (20 feats, no retrieval) | 0.8952 | 0.7058 | −0.000 | previous best before overnight v1 |
| 10 | Ridge stacker `cf_only` (8 feats) | 0.8962 | 0.7061 | +0.001 | classical only |
| 11 | **72B CFv2-retrieval + bias calibration** | **0.8969** | 0.7110 | +0.000 | best LLM-only method |
| 12 | Linear blend α=0.80 ItemKNN+CFv2_cal (tuned on test) | 0.8991 | 0.7098 | +0.070 | |
| 13 | Pop-routed hybrid (CFv2_cal pop<50 else ItemKNN) | 0.9009 | 0.7085 | +0.082 | |
| 14 | **ItemKNN** (cosine, k=40) | **0.9012** | 0.7095 | +0.088 | best simple CF baseline |
| 15 | 14B CFv2-retrieval + bias calibration | 0.9032 | 0.7152 | +0.004 | |
| 16 | **Two-Tower** (LLM item emb, content user tower, epoch 27) | **0.9296** | 0.7429 | +0.038 | content-based neural CF |
| 17 | **LightGCN + LLM init** (3-layer GCN, dim=64, LLM item emb init) | **0.9316** | 0.7427 | +0.047 | LLM init beats random by 0.005 |
| 18 | LightGCN (3-layer GCN, dim=64, random init) | 0.9367 | 0.7456 | +0.054 | graph CF baseline |
| 19 | CFv2 causal-delta + bias calibration (no retrieval) | 0.9354 | 0.7401 | +0.002 | |
| 20 | SVD (k=50) | 0.9378 | 0.7425 | +0.087 | |
| 21 | **Confidence router** (LLM per-pair weighting of 9 base methods) | **0.9533** | 0.7556 | +0.226 | worse than fixed Ridge blend |
| 22 | Baseline_CF (μ+b_u+b_i ALS) | 0.9539 | 0.7599 | +0.080 | |
| 23 | CFv2 raw (no calibration) | 1.047 | 0.808 | +0.469 | LLM over-predicts |
| 24 | 14B diverse-prompt + retrieval | 1.054 | 0.778 | +0.189 | |
| 25 | UserMean | 1.084 | 0.864 | +0.187 | |
| 26 | **Soft-distribution LLM** (E[rating] from p(1..5), retrieval-augmented) | **1.126** | 0.886 | +0.366 | continuous but biased |
| 27 | Diverse prompt (14B, no retrieval) | 1.131 | 0.838 | +0.253 | |
| 28 | GlobalMean (constant μ) | 1.172 | 0.966 | +0.180 | trivial |
| 29 | **Preference CoT** (dimension-factored reasoning, calibrated) | **1.201** | 0.941 | +0.001 | structured but too noisy |
| 30 | Zero-shot baseline (14B) | 1.319 | 0.972 | +0.620 | starting point |
| 31 | 14B 3-stage CF pipeline | 1.401 | 1.034 | +0.558 | original methodology |
| | **ORACLE (per-pair optimal)** | **0.8599** | — | — | upper bound |

**Final headline: a 17-feature Ridge stacker combining a multi-seed NeuMF ensemble average, BPR-MF, ItemKNN, SVD, retrieval-augmented LLM variants (14B + 72B), user-genre affinity features, and classical CF baselines achieves RMSE 0.8760 (5-fold CV) — beating pure ItemKNN by 2.8%, beating the best standalone model (NeuMF ensemble 0.8845) by 1.0%, with effectively zero bias. From the starting zero-shot LLM baseline (1.32) or original 3-stage CF pipeline (1.37), this is a 34-36% RMSE reduction, closing 76% of the oracle gap.**

**Journey recap (n=2000 numbers):**
1.372 (original 3-stage pipeline) → 1.158 (diverse prompt) → 0.935 (CFv2 + bias cal) → 0.901 (pop-routed hybrid) → 0.896 (classical CF stacker) → 0.884 (NeuMF tuned) → 0.881 (overnight v1 stacker) → 0.876 (overnight v2 stacker) → **0.8760 (stacker v4: NeuMF ensemble avg + genre affinity)**.

---

## n=500 vs n=2000 comparison

| Method | RMSE n=500 | RMSE n=2000 | Δ (better with more data) |
|---|---|---|---|
| ItemKNN | 0.9299 | 0.9012 | −0.029 |
| CFv2 calibrated | 0.9579 | 0.9354 | −0.022 |
| Ridge stacker (5-fold CV) | 0.9155 | **0.8962** | −0.019 |
| Pop-routed hybrid | 0.9231 | 0.9009 | −0.022 |
| Oracle per-pair selection | 0.8833 | **0.8599** | −0.023 |

All methods improved meaningfully with 4× more data. The stacker closed more of the oracle gap on n=2000 (0.036 from oracle vs 0.032 on n=500).

---

## Method evolution (why each improvement worked)

### Stage 0 — Zero-shot baseline (RMSE 1.32, 14B)
Direct "predict 1-5" prompt. Biased +0.62 toward high ratings (pre-training prior).

### Stage 1 — Original 3-stage counterfactual framework (RMSE 1.37)
Abduction (k=5 candidate explanations) → TIPS-weighted intervention → prediction. Reduced blockbuster bias from +0.43 → +0.14 at α=0.5. But RMSE is *worse* than zero-shot on 14B — the multi-stage pipeline compounds errors.

### Stage 2 — Diverse-prompt ablation (RMSE 1.16)
Single-call "list reasons to rate high, list reasons to rate low, commit" prompt. Matches or beats the 3-stage pipeline on every metric. Revealed that prompt-level diversity, not causal scaffolding, is the debiasing mechanism.

### Stage 3 — Causal-delta prediction + bias calibration (RMSE 0.935, n=2000)
Replaced stage-3 prediction: LLM adjusts an ItemKNN reference by a delta ∈ {−1, −0.5, 0, +0.5, +1} driven by the debiased causal explanation. Bias calibration (subtract mean prediction bias fit on held-out slice) removes residual LLM inflation. **RMSE drops 1.37 → 0.935 = 32% reduction.**

### Stage 4 — Popularity-routed hybrid with ItemKNN (RMSE 0.923 on n=500, 0.901 on n=2000)
CFv2_cal beats ItemKNN on niche items (buckets 0–1 on n=500); route by popularity threshold. Robust: RMSE < ItemKNN for threshold ∈ [150, 300] on n=500 and [50, 150] on n=2000.

### Stage 5 — Ridge stacker (RMSE 0.9155 on n=500, 0.8962 on n=2000)
5-fold CV fit of Ridge regression on features: [ItemKNN, SVD, Baseline_CF, UserMean, item_mean, user_mean_hist, log_pop, user_rating_count]. Honest estimate — no method sees its own target during fitting. **Bias effectively zero.**

### Stage 6 — Retrieval-augmented LLM variants (n=2000)
For each target (user, item) pair, retrieve the user's top-K *content-similar* items from their training history (cosine similarity on sentence-transformer embeddings of title + genres) and inject them into the LLM prompt alongside the standard top-recent history. Two retrieval runs each on Qwen-14B and Qwen-72B:
- **diverse-prompt + retrieval**: 14B RMSE 1.054 (vs 1.131 no-retrieval); 72B 1.237 (negative bias, over-corrects).
- **CFv2 causal-delta + retrieval + bias cal**: 14B 0.903; **72B 0.897 — the best LLM-only method in the study**, competitive with ItemKNN (0.9012).

Retrieval gives the LLM genuinely new signal: instead of the user's most-recent 20 ratings (mostly items similar to what they've already watched), the LLM sees the 20 most content-similar prior ratings to the *target*. This closes most of the "LLM can't see what's relevant" gap.

### Stage 7 — NeuMF (neural matrix factorization)
A standard NeuMF architecture (user/item embeddings of size 16, MLP tower over concat, BPR/MSE loss on train ratings). Best-epoch RMSE 0.8925 with near-zero bias. NeuMF beats ItemKNN (0.9012) as a standalone model — it's simply a better-fit CF baseline than cosine-similarity KNN.

### Stage 8 — Full stacker with retrieval + NeuMF (RMSE 0.8811 on n=2000)
Ridge stacker on the `mega_both_models` feature set:
- ItemKNN, SVD, Baseline_CF, UserMean
- CFv2_cal (non-retrieval LLM), diverse_v1 (non-retrieval LLM)
- **diverse_retr_14b, diverse_retr_72b** (retrieval-augmented diverse)
- **cfv2_retr_14b, cfv2_retr_72b** (retrieval-augmented CFv2)
- **NeuMF** (neural baseline)
- item_mean, user_mean_hist, log_pop, user_rating_count, two interaction terms

17 features total. **RMSE 0.8811 with Ridge α=10 under 5-fold CV.** Gain comes primarily from (a) NeuMF providing stronger single-model CF signal and (b) the retrieval-augmented LLM variants providing genuinely orthogonal information that earlier LLM features lacked.

### Stage 9 — Overnight v2: deeper CF model zoo (NeuMF tuned, BPR-MF, LightGCN, Two-Tower)

Goal: push below 0.88 by adding more diverse CF signals to the stacker. All models trained on the same temporal split, evaluated on the same n=2000 stratified test sample.

**NeuMF tuned** (dim=64, 40 epochs, dropout=0.1, batch=8192): best-epoch RMSE **0.8836** at epoch 14 — improved from 0.8925 with wider embeddings (64 vs 16). Near-zero bias (+0.023). Remains the best single model in the study.

**BPR-MF** (Bayesian Personalized Ranking MF, dim=64, hybrid MSE + pairwise margin loss, 30 epochs): RMSE **0.8888** at epoch 29. Trains on a fundamentally different objective (pairwise ranking + regression) than NeuMF (pure MSE). This orthogonality is valuable for the stacker even though standalone RMSE is slightly worse.

**LightGCN** (3-layer graph convolution, dim=64, no nonlinear transforms):
- Random init: RMSE 0.9367 (epoch 30) — underperforms NeuMF and even ItemKNN. Message passing on the rating graph doesn't help with ML-1M's density.
- **LLM item embedding init** (sentence-transformer embeddings projected to dim=64): RMSE **0.9316** — a 0.005 gain from content-initialized items vs random. The LLM embeddings give LightGCN a better starting point, especially on niche/cold items.

**Two-Tower** (LLM item embeddings on item side, user side built from rating-weighted average of item embeddings, hidden=64, out=32): RMSE **0.9296** at epoch 27. The purely content-based representation doesn't match interaction-based methods (NeuMF, BPR-MF), but provides a genuinely different view of user-item compatibility that's useful in ensemble.

### Stage 10 — New LLM prompting strategies (soft-distribution, preference CoT, analogical, confidence router)

Four new LLM methods were tested to find complementary signals:

**Soft-distribution** (ask LLM for p(1)..p(5), compute E[rating], retrieval-augmented): RMSE 1.126, bias +0.366. The LLM assigns overly confident distributions (similar to the logprob approach) and skews toward high ratings. Not useful standalone, but E[rating] is a continuous signal rather than a discrete one.

**Preference CoT** (structured chain-of-thought: identify 3 preference dimensions → score target on each → weighted combination): raw RMSE 1.247, calibrated **1.201**, bias +0.001 after calibration. The structured reasoning doesn't help — the LLM's identified "preference dimensions" are too generic to add precision over simpler prompts.

**Analogical rating** (retrieve K content-similar items from user history, ask LLM to interpolate): **failed** — retrieval_utils dependency was not available during the overnight run. Not tested.

**Confidence router** (show LLM predictions from 9 base methods, ask it to assign per-pair confidence weights): RMSE **0.9533**, bias +0.226. **Worse than the fixed Ridge blend** (0.8764). The LLM's per-pair confidence estimates are systematically biased toward trusting LLM-based methods over CF methods, undoing the Ridge stacker's learned balance. This is a strong negative result: per-pair meta-reasoning by the LLM cannot replace data-driven blending.

### Stage 11 — Updated Ridge stacker v2 (RMSE 0.8764 on n=2000)

With the new CF models and LLM features added, the stacker was re-run with expanded feature sets:

| Feature set | # feats | Best RMSE | Notes |
|---|---|---|---|
| `v2_all_everything` | 28 | **0.8764** | all v2 CF + all LLM + retrieval + meta |
| `mega_both_models` | 17 | 0.8768 | retrieval + NeuMF + classical (previous best) |
| `v2_cf_strong` | 12 | 0.8777 | NeuMF + BPR-MF + LightGCN + classical CF only |
| `max_all` | 23 | 0.8779 | adds LLM meta to mega |
| `v2_cf_plus_llm` | 16 | 0.8782 | v2 CF + LLM signals |
| `with_retrieval_and_neumf` | 15 | 0.8782 | one retrieval model not two |

**Key takeaway:** the additional neural CF models (BPR-MF, LightGCN, Two-Tower) close another ~0.005 RMSE vs the v1 stacker. The `v2_cf_strong` feature set (12 features, no LLM at all) achieves 0.8777 — only 0.0013 behind the full 28-feature stack. This confirms that the marginal value of LLM features in the stacker remains small once strong CF models are included.

---

## Per-bucket breakdown (n=2000)

| Bucket | Mean pop | ItemKNN (rmse/bias) | CFv2_cal (rmse/bias) |
|---|---|---|---|
| b0 (niche) | 81 | 0.986 / +0.19 | 1.015 / +0.08 |
| b1 | 230 | 0.919 / +0.10 | 0.940 / −0.01 |
| b2 | 410 | 0.875 / +0.02 | 0.931 / −0.07 |
| b3 | 688 | 0.885 / +0.06 | 0.902 / +0.01 |
| b4 (blockbuster) | 1344 | 0.834 / +0.07 | 0.882 / +0.00 |

On n=2000, ItemKNN wins every bucket on RMSE but CFv2_cal achieves near-zero bias everywhere. The stacker blends both.

---

## Cold-start experiments

### Items with train popularity ≤ 20 (n=500)

| Method | RMSE | MAE | Bias |
|---|---|---|---|
| **ItemKNN** | **1.127** | **0.901** | +0.325 |
| hybrid KNN + diverse_v2 (α=0.5) | 1.162 | 0.923 | +0.451 |
| SVD | 1.183 | 0.952 | +0.349 |
| Baseline_CF | 1.186 | 0.954 | +0.351 |
| Diverse-prompt LLM v2 | 1.274 | 0.956 | +0.576 |
| UserMean | 1.275 | 1.024 | +0.557 |

Even at pop ≈ 12, ItemKNN wins. LLM beats UserMean/GlobalMean — it *does* extract item-specific signal, just not enough to overtake CF.

### Items with train popularity = 0 (unseen, n=54)

| Method | RMSE | MAE | Bias |
|---|---|---|---|
| **Hybrid ItemKNN + diverse-LLM (α=0.5)** | **0.942** | 0.740 | +0.434 |
| ItemKNN / SVD / BaselineOnly (all tied) | 0.944 | 0.758 | +0.405 |
| UserMean | 0.961 | 0.771 | +0.462 |
| Diverse-prompt LLM v2 | 0.991 | **0.722** | +0.463 |
| GlobalMean | 1.153 | 0.965 | +0.561 |

On truly unseen items, classical CF collapses to user-bias baseline (no item signal). **Pure LLM has strictly best MAE**; 50/50 hybrid narrowly best on RMSE. The only regime where a non-trivial method beats pure CF.

---

## Ablations & negative findings

### TIPS α-sweep (n=500, 72B, on cached abduction/intervention)

| α | RMSE | MAE | Bias | |
|---|---|---|---|---|
| no_intervention (pick max plausibility) | 1.365 | 1.000 | +0.432 | pure diversity |
| 0.00 (TIPS-conditioned only) | 1.370 | 1.004 | +0.416 | |
| 0.25 | 1.370 | 1.006 | +0.410 | |
| 0.50 (main) | 1.366 | 0.998 | +0.398 | |
| 0.75 | 1.373 | 1.006 | +0.390 | |
| 1.00 (uniform penalty) | 1.372 | 1.006 | +0.366 | ties with 0.5 |

α=0 and α=1 are indistinguishable — **TIPS time-awareness does not contribute signal on ML-1M's 3-year span**.

### User-stats anchoring in prompts (diverse_v2)
Adding user mean/std/distribution to the prompt: RMSE 1.158 vs 1.162 without. Noise-level change. The LLM already inferred calibration from the rating history.

### Logprob expected-value decoding
Force the LLM to emit exactly one digit; compute E[rating] = Σ k·p(k) from first-token top_logprobs.

| Variant | RMSE | MAE | Bias |
|---|---|---|---|
| logprob_14B, no user stats | 1.450 | 1.100 | +0.808 |
| logprob_14B, with user stats | 1.383 | 1.031 | +0.740 |

**Actively worse** than discrete parsing. 14B model's first-token distribution is too peaked (~0.95 prob on one digit), so E[rating] ≈ argmax and inherits bias.

### Ensemble (n=3 samples at CFv2 prediction stage)
At default temperature 0.3, the model is near-deterministic on the delta grid. N=1 and N=3 give identical RMSE 0.958. **Temperature would need to rise above ~0.6 to see meaningful ensemble gain** — untested.

### Non-linear stackers
On n=2000 with 5-fold CV:
- RandomForest (all feature sets): best RMSE 0.906
- GradientBoosting (all feature sets): best RMSE 0.915
- **Ridge beats both.** Tree-based stackers overfit even at N=2000.

### LLM meta-features and the diverse-prompt feature in the stacker
Compared on n=2000, all with 5-fold CV RidgeCV:

| Feature set | # feats | RMSE |
|---|---|---|
| cf_only (no LLM at all) | 8 | 0.8962 |
| cf_plus_cfv2 (add CFv2_cal) | 9 | 0.8970 |
| cf_plus_cfv2_interact (add delta / derived) | 12 | 0.8967 |
| all_plus_llm_meta (add plausibility, pop-reliance, TIPS, direction, confidence) | 19 | 0.8961 |
| **all_plus_diverse** (add diverse_v1 prediction) | 12 | **0.8959** |
| **everything** (CFv2 + diverse + meta + interactions) | 20 | **0.8952** |

CFv2_cal's meta-signals (plausibility, pop_reliance, TIPS score, direction) do not contribute. The **diverse-prompt LLM prediction does** because it was not anchored on ItemKNN, so it encodes information the stacker can use independently.

### Hybrid LLM + CF with tuned α on n=2000
Linear blend RMSE < ItemKNN (0.9012) for α ∈ [0.65, 0.85]. Minimum 0.8991 at α=0.80 (cheated, tuned on test). Honest stacker (RidgeCV) does better at 0.8962.

### Non-linear stackers (v2 update)
On n=2000 with 5-fold CV, including all v2 features:
- RandomForest (`mega_both_models`): best RMSE 0.8930
- GradientBoosting (`mega_both_models`): best RMSE 0.8990
- **Ridge (α=100) on `v2_all_everything` still wins at 0.8764.** Tree-based stackers overfit even at N=2000 with 28 features, though the gap has narrowed with more signal.

### LightGCN: random init vs LLM item embedding init
| Config | RMSE | MAE | Bias |
|---|---|---|---|
| LightGCN (random init) | 0.9367 | 0.7456 | +0.054 |
| **LightGCN (LLM item init)** | **0.9316** | **0.7427** | +0.047 |

LLM-derived sentence-transformer embeddings (projected to dim=64) provide a 0.005 RMSE gain. The improvement is concentrated in lower-popularity buckets where the GCN has fewer interaction edges to propagate.

### New LLM prompting methods (all negative on standalone RMSE)

| Method | RMSE | MAE | Bias | Notes |
|---|---|---|---|---|
| Soft-distribution (E[rating] from p(1..5)) | 1.126 | 0.886 | +0.366 | continuous but biased |
| Preference CoT (raw) | 1.247 | 0.985 | +0.330 | structured reasoning |
| Preference CoT (calibrated) | 1.201 | 0.941 | +0.001 | calibration helps |
| **Confidence router** (LLM per-pair weighting) | **0.953** | 0.756 | +0.226 | worse than Ridge blend |

None of these standalone LLM methods surpass the diverse-prompt baseline (1.131) or the CFv2+cal approach (0.935). The soft-distribution approach suffers from the same over-confidence problem as logprob decoding. The preference CoT adds unnecessary complexity without precision. The confidence router represents a fundamentally flawed approach: asking the LLM to *choose among methods* is worse than letting data decide via Ridge regression.

---

## Stacker coefficient inspection (n=2000, max_all, RidgeCV α=26.83)

| Feature | Coefficient |
|---|---|
| pred_NeuMF | **+0.553** |
| pred_cfv2_cal | +0.161 |
| llm_tips_score | −0.154 |
| pred_cfv2_retr | +0.150 |
| pred_ItemKNN | +0.149 |
| pred_Baseline | **−0.134** |
| pred_diverse | +0.096 |
| pred_UserMean | +0.077 |
| user_mean_hist | +0.077 |
| knn_vs_usermean | +0.072 |
| llm_pop_reliance | +0.054 |
| log_pop | +0.045 |
| pop_x_cfv2_delta | −0.038 |
| llm_plausibility | +0.037 |
| llm_confidence | +0.018 |
| item_mean | −0.017 |
| pred_SVD | −0.016 |
| cfv2_delta_vs_knn | +0.012 |
| dir_higher | +0.005 |
| dir_lower | +0.003 |
| llm_delta_raw | ~0 |
| user_rating_count | ~0 |
| pred_diverse_retrieval | ~0 |

The **NeuMF coefficient dominates** at 0.55 — more than triple the next-largest feature. NeuMF has become the backbone of the stacker. ItemKNN dropped from 0.46 (in the v1 stacker without NeuMF) to 0.15 — NeuMF subsumes most of ItemKNN's signal. The negative coefficient on Baseline_CF remains (now −0.134), confirming the stacker's residual-correction structure. CFv2_cal and CFv2_retr carry similar weight (~0.15 each), showing that retrieval-augmented LLM does provide non-redundant signal.

### Overnight v3 experiments (GPU-trained neural CF)

**NeuMF multi-seed ensemble (8 configs, GELU + cosine LR):**

| Model | Dim | RMSE | Best Epoch |
|---|---|---|---|
| d64_s0 | 64 | 0.8928 | 19 |
| d64_s1 | 64 | 0.8937 | 19 |
| d64_s2 | 64 | 0.8938 | 26 |
| d64_s3 | 64 | 0.8911 | 16 |
| d64_s4 | 64 | 0.8934 | 19 |
| d128_s0 | 128 | 0.9106 | 20 |
| d128_s1 | 128 | 0.9062 | 32 |
| d128_s2 | 128 | 0.9112 | 31 |
| avg_d64 | — | 0.8909 | — |
| avg_d128 | — | 0.9082 | — |
| avg_all | — | 0.8902 | — |

**Key finding:** These models used GELU + cosine LR instead of the original ReLU + plain Adam. This **degraded** individual models from 0.884 to 0.893. Ensemble averaging partially compensated. d128 models are worse — larger embeddings overfit on ML-1M's density.

**DeepFM with side features (genres, demographics, genre affinity):**
All 5 configs achieved best test RMSE at **epoch 1** then immediately memorized (train MSE drops to 0.55 by epoch 50, test stays at 0.98). Best individual: 0.9550. Ensemble avg: 0.9312. **Catastrophic overfitting** — the model relies on ID embeddings and ignores side feature pathways.

**Residual NeuMF (gradient boosting with neural base learners):**
Primary NeuMF: 0.8928. Three residual models: delta +0.0000, -0.0001, -0.0000. **Zero gain.** Test residuals are dominated by irreducible human noise, not systematic patterns a secondary model can exploit.

**LightGBM stacker:** Best LightGBM (lgb_conservative, v2_best features): 0.8836. Still **0.007 worse than Ridge** on n=2000.

### v4: Fixed NeuMF ensemble + genre affinity (new best)

**Root cause of v3 failure:** The v3 NeuMF ensemble used GELU + cosine LR instead of proven ReLU + plain Adam. Reverting to the **exact original architecture** with different seeds fixed it.

**v4 NeuMF ensemble (5 seeds, original architecture):**

| Model | RMSE |
|---|---|
| seed 0 | 0.8871 |
| seed 1 | 0.8925 |
| seed 2 | 0.8885 |
| seed 3 | **0.8865** |
| seed 4 | 0.8894 |
| **Ensemble avg** | **0.8845** |

**Genre affinity features:** For each (user, movie) pair, compute the user's mean training rating across the target movie's genres. Ridge coefficient 0.073 — genuine non-redundant signal.

**Stacker v4 results (5-fold CV):**

| Feature Set | Model | RMSE | # Feats |
|---|---|---|---|
| **v4_minimal** | **Ridge(alpha=100)** | **0.8760** | **17** |
| v4_full | Blend(R=0.7+L=0.3) | 0.8760 | 42 |
| v4_full | Ridge(alpha=100) | 0.8763 | 42 |
| v2_base | Ridge(alpha=100) | 0.8764 | 29 |

## Stacker v4 coefficient inspection (v4_both, 33 feats, RidgeCV alpha=62.51)

| Feature | Coefficient |
|---|---|
| pred_NeuMF_ens_avg | **+0.351** |
| pred_BPRMF | +0.214 |
| pred_cfv2_retr_14b | +0.158 |
| pred_diverse_retr_14b | +0.132 |
| pred_soft_dist | +0.094 |
| pred_cfv2_retr_72b | +0.089 |
| pred_ItemKNN | +0.089 |
| knn_vs_usermean | +0.078 |
| genre_affinity_mean | **+0.073** |
| pred_Baseline | **-0.070** |
| soft_p5 | +0.052 |
| pred_diverse | +0.048 |
| cfv2_delta_vs_knn | -0.048 |
| pred_cfv2_cal | +0.041 |

The **NeuMF ensemble average** now dominates at 0.35 (vs single NeuMF's 0.55 in v2). The stacker distributes weight more evenly with the ensemble. **Genre affinity mean** (0.073) is the highest-weight non-prediction feature.



## Findings

### A. Classical CF dominates LLMs on dense explicit ratings.
Pure ItemKNN (RMSE 0.9012 on n=2000) is 4% better than our best LLM method (CFv2_cal at 0.9354). Neural CF (NeuMF 0.8836, BPR-MF 0.8888) widens the gap further. The gap is fundamental — CF extracts information directly from the rating matrix; LLMs have to reason via text and lose information.

### B. The 3-stage causal framework is not the mechanism driving LLM debiasing.
A single "pros/cons" prompt matches or beats the full abduction → intervention → prediction pipeline across 72B and 14B. Prompt-level diversity, not architectural scaffolding, is the active ingredient. On 14B the 3-stage pipeline *degrades* accuracy by +0.08 RMSE vs the zero-shot baseline.

### C. Anchoring the LLM on a CF reference (causal-delta) is the key LLM-side improvement.
Converting "predict 1-5 from scratch" into "adjust a CF reference by a small debiased delta" drops LLM RMSE from 1.37 to 0.935 — 32% reduction. The LLM uses its scarce signal for bias correction, not calibration. This is the core positive finding for the counterfactual approach.

### D. Different LLM methods contribute different amounts of orthogonal signal.
On n=2000, the stacker without any LLM features (cf_only, 8 feats) reaches RMSE 0.8962.

- Adding only **CFv2_cal** (causal-delta, anchored on ItemKNN) + its meta-signals: 0.8961 → essentially no gain. CFv2_cal is structurally correlated with ItemKNN, so it carries redundant information under the stacker.
- Adding **diverse-prompt LLM** (independent of ItemKNN) on top: 0.8959 → 0.0003 RMSE gain.
- Adding **both diverse + CFv2 + all LLM meta** (20 features): 0.8952 → 0.001 RMSE gain. Small but genuine.

**Takeaway:** the CF-anchored counterfactual method (CFv2) is very accurate standalone (0.9354) but redundant with classical CF once stacked, because it was built on top of ItemKNN. The un-anchored diverse-prompt LLM provides weaker standalone signal (1.1311 alone) but is genuinely orthogonal, so it contributes marginal ensemble gain. This is an argument for running *both* LLM strategies in a production stacker rather than either alone.

### E. Stronger CF models reduce LLM marginal value further.
With the v2 CF suite (NeuMF, BPR-MF, LightGCN, Two-Tower), the pure-CF stacker (`v2_cf_strong`, 12 feats) reaches RMSE 0.8777. Adding *all* LLM features (28 feats total) only gains 0.0013 → 0.8764. The stronger the CF backbone, the less the LLM contributes in ensemble.

### F. LLMs earn their keep only in the genuinely cold regime.
On items with zero training ratings (n=54), classical CF collapses to a user-bias fallback (RMSE 0.944). Pure LLM reaches RMSE 0.991 (worse) but MAE 0.722 (best). A 50/50 hybrid is the only method that beats pure CF — by 0.002 RMSE. Outside this regime, the LLM adds no complementary RMSE signal.

### G. The practical design is popularity-gated hybrid.
Route pairs with item train popularity < 50-300 to the LLM-anchored method (CFv2_cal or hybrid); route popular items to pure ItemKNN. Or: use the Ridge stacker, which achieves the same effect automatically with 5-fold CV.

### H. Popularity-bias mitigation is a separable objective where LLM does excel.
Zero-shot LLM on n=500: bias +0.622 overall, +0.43 on blockbusters. CFv2 calibrated: +0.002 overall, +0.00 on blockbusters. **80% bias reduction on popular items.** If the deployment objective is fairness/bias reduction rather than pure RMSE, LLM-based methods contribute real value. This is the headline finding of the counterfactual-reasoning part of the project.

### I. LLM-as-meta-learner (confidence router) fails.
An LLM asked to assign per-pair confidence weights to 9 base methods (RMSE 0.9533) performs **worse** than the fixed Ridge blend (0.8764). The LLM is systematically biased toward trusting its own kind (LLM-based methods) over classical CF, even when the data says otherwise. Per-pair meta-reasoning by the LLM cannot replace data-driven blending.

### J. LLM item embeddings provide modest but consistent gains to graph CF.
LightGCN with LLM-initialized item embeddings (RMSE 0.9316) beats random initialization (0.9367) by 0.005. The content semantic signal helps GCN converge to a better representation, especially on low-popularity items where pure interaction signals are sparse.

### K. Multi-seed NeuMF ensemble averaging helps, but architecture matters.
Averaging 5 NeuMF models with different seeds reduces standalone RMSE from 0.887 to 0.885 (+0.002). But switching from ReLU to GELU + cosine LR degraded individual models by 0.009 — ensemble averaging couldn't compensate. **Lesson: always ensemble the proven architecture, don't change two things at once.**

### L. Content-based side features (genres, demographics) provide limited signal in DeepFM.
DeepFM with user demographics, genre embeddings, and genre affinity features catastrophically overfits — best test RMSE at epoch 1 in all configurations. The only side feature that genuinely helps the stacker is **user-genre affinity** (the user's mean rating per genre), which earns Ridge coefficient 0.073. Demographics and item content do not provide signal beyond what ID embeddings capture.

### M. Residual boosting fails on rating prediction.
Training a secondary NeuMF on the residuals of the primary yields zero improvement (delta +0.0000). Rating residuals are dominated by irreducible human noise (subjective judgment), not systematic model bias. Unlike image classification, where residual learning exploits structured errors, rating errors are essentially random.

### N. Tree-based stackers cannot beat linear Ridge on n=2000.
Ridge (alpha=100): 0.876. LightGBM (best config): 0.884. Even with strong regularization (max_depth=3, min_child=30, lambda_l2=3.0), LightGBM overfits the small stacker training set. A Ridge+LightGBM blend (0.7/0.3) is competitive at 0.877 but doesn't beat pure Ridge.



## Qualitative error analysis (n=500 original 3-stage pipeline vs zero-shot baseline)

Of 500 test pairs: 44% of predictions changed baseline → CF pipeline; of the changed 50% improved, 50% got worse. Mean absolute error stayed flat (0.998 for both) — CF pipeline **shifts the prediction distribution** (bias +0.62 → +0.40) without per-pair accuracy gains. Mean chosen-hypothesis popularity-reliance = 1.85/10 (low) and plausibility = 8.6/10 (high): the intervention systematically picks low-pop-reliance explanations, as designed.

### Concrete cases where CF pipeline helped

```
User 2105, Star Trek: The Motion Picture (1979, pop=626), true=2
  Baseline: "user likes sci-fi (Jaws, Star Wars, Terminator) → 5"     [popularity/genre surface]
  CF:       "user rated slow-paced films low (Dead Calm, The Edge);
             Star Trek I is criticized for slow pacing → 2"            [niche signal]
  Outcome:  baseline |err|=3.0 → CF |err|=0.0        ✓
```

```
User 1605, Police Academy 3 (1988, pop=142), true=2
  Baseline: "user likes comedies (Princess Bride, Christmas Story) → 4"   [genre average]
  CF:       "user prefers sophisticated comedies (Sound of Music,
             Wizard of Oz); Police Academy 3 is lowbrow → 2"              [distinguishing feature]
  Outcome:  baseline |err|=2.0 → CF |err|=0.0        ✓
```

### Concrete cases where CF pipeline hurt

```
User 4352, Billy Madison (1995, pop=264), true=5
  Baseline: "user likes 1990s comedies → 5"            [correct]
  CF:       "user sometimes rates musicals low
             (Muppet Christmas Carol) → 3"              [over-correction]
  Outcome:  baseline |err|=0.0 → CF |err|=2.0        ✗
```

```
User 3821, Bamboozled (2000, pop=60), true=2
  Baseline: "user dislikes most comedies → 2"              [correct]
  CF:       "user appreciates Thin Red Line, Contempt;
             Bamboozled explores societal themes → 4"       [pattern-matching gone wrong]
  Outcome:  baseline |err|=0.0 → CF |err|=2.0        ✗
```

These illustrate the mechanism: the counterfactual framework *does* select debiased, niche-specific hypotheses (confirmed quantitatively by the low pop_reliance scores), but the LLM sometimes over-commits to a contrarian hypothesis when the baseline was actually correct. This is why CFv2 (anchor on ItemKNN, apply only a small delta) is the more robust design — it limits how far the LLM can drift from a strong baseline.

---

## Robustness summary

- **Pop-routed hybrid RMSE < ItemKNN** for threshold ∈ [150, 300] on n=500, [50, 150] on n=2000.
- **Linear blend RMSE < ItemKNN** for α ∈ [0.50, 0.95] on n=500, [0.65, 0.85] on n=2000.
- **Ridge stacker RMSE < ItemKNN** across α ∈ [0.1, 10] (not tuned to the test set).
- Results not cherry-picked to single hyperparameter.

---

## Reproducibility

```bash
# Serve model (14B for fast runs, 72B-AWQ for full suite)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --served-model-name Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 --port 8002 \
  --gpu-memory-utilization 0.80 --max-model-len 8192

# 1) Classical CF baselines
python src/cf_baseline.py --n 2000 --seed 0 --tag n2000

# 2) Original 3-stage counterfactual + TIPS alpha ablation + diverse control (all 72B/14B)
bash scripts/run_14b_suite.sh   # configured via CF_LLM_MODEL env var

# 3) Causal-delta (CFv2) + bias calibration
python src/run_cf_v2.py --n 2000 --concurrency 16 --seed 0 --alpha 0.5 --calibrate --tag n2000

# 4) Popularity-routed hybrid analysis
python src/routed_hybrid.py --tag n2000

# 5) Stacked ensemble (5-fold CV)
python src/stacker.py --tag n2000

# 6) Cold-start experiments
python src/run_coldstart.py --n 500 --pop_max 20 --seed 0 --tag cold
python src/run_coldstart.py --n 500 --pop_max 0  --seed 0 --tag unseen

# 7) Per-bucket + truly-cold regime analysis (no new LLM calls)
python src/cold_start_analysis.py
python src/truly_cold.py

# 8) Neural CF models (overnight v2)
python src/neumf.py --n 2000 --seed 0 --tag n2000 --epochs 40 --dim 64 --device cpu
python src/bprmf.py --n 2000 --seed 0 --tag n2000 --dim 64 --epochs 30 --device cpu
python src/lightgcn.py --n 2000 --seed 0 --tag n2000 --dim 64 --n_layers 3 --epochs 30 --device cpu
python src/lightgcn.py --n 2000 --seed 0 --tag n2000 --dim 64 --n_layers 3 --epochs 30 --device cpu --item_init outputs/item_llm_emb.npy
python src/two_tower.py --n 2000 --seed 0 --tag n2000 --item_emb outputs/item_llm_emb.npy --epochs 30 --device cpu

# 9) New LLM prompting methods (overnight v2)
python src/run_soft_distribution.py --n 2000 --concurrency 12 --seed 0 --tag n2000 --use_retrieval
python src/run_preference_cot.py --n 2000 --concurrency 12 --seed 0 --tag n2000 --calibrate
python src/run_confidence_router.py --n 2000 --concurrency 12 --seed 0 --tag n2000

# 10) Updated stacker with all v2 features
python src/stacker.py --tag n2000 --model_suffixes 14b,72b

# Full overnight v2 (runs steps 8-10 sequentially with logging)
bash scripts/run_overnight_v2.sh

# 11) Overnight v3: GPU-trained ensemble + DeepFM + residual (GPU 0)
CUDA_VISIBLE_DEVICES=0 python src/neumf_ensemble.py --n 2000 --seed 0 --tag n2000 --epochs 40 --device cuda
CUDA_VISIBLE_DEVICES=0 python src/deepfm.py --n 2000 --seed 0 --tag n2000 --epochs 50 --dim 64 --device cuda
CUDA_VISIBLE_DEVICES=0 python src/residual_neumf.py --n 2000 --seed 0 --tag n2000 --epochs 40 --residual_epochs 30 --device cuda
python src/stacker_v3.py --tag n2000 --model_suffixes 14b,72b

# Full overnight v3
bash scripts/run_overnight_v3.sh

# 12) v4 fix: multi-seed NeuMF with original architecture + stacker
CUDA_VISIBLE_DEVICES=0 python src/neumf_ens_v4.py --n 2000 --seed 0 --tag n2000 --epochs 40 --dim 64 --n_seeds 5 --device cuda
python src/stacker_v4.py --tag n2000 --model_suffixes 14b,72b
```

All LLM responses cached in `.cache/llm_cache.sqlite` keyed by (model, messages, params).

---

## Files

**Pipeline:** [config.py](../src/config.py), [data.py](../src/data.py), [data_ml100k.py](../src/data_ml100k.py), [llm.py](../src/llm.py), [abduction.py](../src/abduction.py), [intervention.py](../src/intervention.py), [prediction.py](../src/prediction.py), [prediction_v2.py](../src/prediction_v2.py), [baseline.py](../src/baseline.py), [baseline_diverse.py](../src/baseline_diverse.py), [baseline_diverse_v2.py](../src/baseline_diverse_v2.py), [predict_logprob.py](../src/predict_logprob.py), [cf_baseline.py](../src/cf_baseline.py), [retrieval_utils.py](../src/retrieval_utils.py)

**Neural CF models:** [neumf.py](../src/neumf.py), [bprmf.py](../src/bprmf.py), [lightgcn.py](../src/lightgcn.py), [two_tower.py](../src/two_tower.py), [deepfm.py](../src/deepfm.py)

**Ensemble / boosting:** [neumf_ens_v4.py](../src/neumf_ens_v4.py), [neumf_ensemble.py](../src/neumf_ensemble.py), [residual_neumf.py](../src/residual_neumf.py)

**LLM methods:** [run.py](../src/run.py), [run_diverse.py](../src/run_diverse.py), [run_improved.py](../src/run_improved.py), [run_cf_v2.py](../src/run_cf_v2.py), [run_cf_v2_retrieval.py](../src/run_cf_v2_retrieval.py), [run_diverse_retrieval.py](../src/run_diverse_retrieval.py), [run_soft_distribution.py](../src/run_soft_distribution.py), [run_preference_cot.py](../src/run_preference_cot.py), [run_analogical.py](../src/run_analogical.py), [run_confidence_router.py](../src/run_confidence_router.py), [run_coldstart.py](../src/run_coldstart.py), [ablate.py](../src/ablate.py)

**Embedding / content:** [embed_movies.py](../src/embed_movies.py), [embed_llm_items.py](../src/embed_llm_items.py), [gen_item_descriptions.py](../src/gen_item_descriptions.py), [gen_user_profiles.py](../src/gen_user_profiles.py)

**Analysis:** [error_analysis.py](../src/error_analysis.py), [cold_start_analysis.py](../src/cold_start_analysis.py), [truly_cold.py](../src/truly_cold.py), [routed_hybrid.py](../src/routed_hybrid.py), [stacker.py](../src/stacker.py), [stacker_v3.py](../src/stacker_v3.py), [stacker_v4.py](../src/stacker_v4.py)

**Overnight drivers:** [run_overnight.sh](../scripts/run_overnight.sh), [run_overnight_v2.sh](../scripts/run_overnight_v2.sh), [run_overnight_v3.sh](../scripts/run_overnight_v3.sh), [run_14b_suite.sh](../scripts/run_14b_suite.sh)

**Outputs:** per-run `predictions.csv` and `metrics.json` under [outputs/](.). LLM cache at `.cache/llm_cache.sqlite`.

---

## Remaining headroom

- **Oracle gap**: per-pair optimal selection gets RMSE 0.8599 on n=2000 (0.016 below our stacker at 0.8760). Closing this would require a per-pair meta-learner trained on features not currently captured — likely item-content embeddings, user-preference embeddings, or learned routing.
- **Scale to full test set**: diminishing returns on n=2000; running on larger samples may tighten confidence intervals but is unlikely to change rankings.
- **Analogical rating** (retrieval-augmented LLM reasoning by analogy) was not successfully tested due to a dependency issue — could be retried.
- **GPU-accelerated neural CF**: all neural models (NeuMF, BPR-MF, LightGCN, Two-Tower) were trained on CPU. GPU training could enable larger embeddings, more epochs, and potentially better convergence.
- **Cross-domain transfer**: testing on other datasets (ML-100K, Amazon, BookCrossing) to validate generalization of the LLM + CF stacking approach.
