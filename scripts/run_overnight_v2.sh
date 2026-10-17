#!/bin/bash
# Overnight v2: push beyond the v1 ceiling (RMSE 0.8811) by adding
#   - Stronger CF: LightGCN, BPR-MF, tuned NeuMF
#   - LLM-enhanced CF: LightGCN with LLM item embedding init, TwoTower
#   - New LLM signals: soft distribution, preference CoT, analogical, confidence router
#   - Richer stacker
#
# All steps logged separately under outputs/overnight_v2_<timestamp>/.
# Predictions saved to standard output dirs consumed by stacker.py.
set -u

cd "$(dirname "$0")/.."   # run from the repository root
# Activate your Python environment (e.g. `conda activate cf`) before running this script.

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=outputs/overnight_v2_${STAMP}
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/master.log"

VLLM_BIN=${VLLM_BIN:-vllm}
M72_PATH=${M72_PATH:-models/qwen2.5-72b-instruct-awq}   # local AWQ weights (or a HF repo id)
M72_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
M14_NAME=Qwen/Qwen2.5-14B-Instruct

TAG=n2000
SEED=0
N=2000
CONC_72=12
CONC_14=16

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
step() { echo -e "\n[$(date +%H:%M:%S)] === $* ===" | tee -a "$MASTER"; }

run_step() {
    local name="$1"; shift
    local cmd="$*"
    local step_log="$LOG_DIR/${name}.log"
    step "START: $name"
    log "  cmd: $cmd"
    bash -c "$cmd" > "$step_log" 2>&1
    local rc=$?
    step "END:   $name (exit $rc)"
    if [ $rc -ne 0 ]; then
        log "  FAILED. tail of $step_log:"
        tail -30 "$step_log" | tee -a "$MASTER"
    else
        tail -8 "$step_log" | tee -a "$MASTER"
    fi
    return $rc
}

wait_for_ready() {
    local timeout=${1:-600}; local waited=0
    while [ $waited -lt $timeout ]; do
        if curl -sf http://localhost:8002/v1/models > /dev/null 2>&1; then
            return 0
        fi
        sleep 5; waited=$((waited + 5))
    done
    return 1
}

current_model() {
    curl -sf http://localhost:8002/v1/models 2>/dev/null | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(d["data"][0]["id"])' 2>/dev/null || echo "none"
}

switch_to_72b() {
    local cur=$(current_model)
    if [ "$cur" = "$M72_NAME" ]; then
        log "  72B already running."; return 0
    fi
    step "Switching vLLM: $cur -> 72B"
    local pid=$(pgrep -f "vllm serve" | head -1)
    if [ -n "$pid" ]; then
        log "  killing vLLM PID $pid ..."
        kill "$pid"; sleep 10
        if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid"; sleep 3; fi
    fi
    sleep 20
    log "  launching 72B server ..."
    CUDA_VISIBLE_DEVICES=1 nohup "$VLLM_BIN" serve "$M72_PATH" \
        --served-model-name "$M72_NAME" \
        --host 127.0.0.1 --port 8002 \
        --gpu-memory-utilization 0.90 --max-model-len 8192 \
        > "$LOG_DIR/vllm_72b_server.log" 2>&1 &
    log "  waiting for 72B readiness ..."
    if wait_for_ready 600; then log "  72B ready."; return 0
    else log "  72B failed to start."; return 1; fi
}

log "Overnight v2 started: $STAMP  TAG=$TAG  N=$N"

# =====================================================================
# Phase A: LLM-generated content (one-time, under whichever model is serving)
# =====================================================================

# Item descriptions (one call per movie, ~3883 calls)
run_step "01_item_descriptions" \
    "python -u src/gen_item_descriptions.py --concurrency 16"

# Item embeddings (CPU, sentence-transformers or TF-IDF fallback)
run_step "02_item_llm_embeddings" \
    "python -u src/embed_llm_items.py"

# User profiles (one call per user, ~6040 calls)
run_step "03_user_profiles" \
    "python -u src/gen_user_profiles.py --concurrency 16"

# =====================================================================
# Phase B: Classical / neural CF models (CPU-friendly; run in parallel)
# =====================================================================

# Tuned NeuMF grid (foreground, CPU)
run_step "04_neumf_tuned" \
    "python -u src/neumf.py --n $N --seed $SEED --tag $TAG --epochs 40 --dim 64 --device cpu"

# BPR-MF (CPU)
run_step "05_bprmf" \
    "python -u src/bprmf.py --n $N --seed $SEED --tag $TAG --dim 64 --epochs 30 --device cpu"

# LightGCN (CPU, can be slow — 30 epochs)
run_step "06_lightgcn" \
    "python -u src/lightgcn.py --n $N --seed $SEED --tag $TAG --dim 64 --n_layers 3 --epochs 30 --device cpu"

# LightGCN with LLM item embedding init
run_step "07_lightgcn_llminit" \
    "python -u src/lightgcn.py --n $N --seed $SEED --tag $TAG --dim 64 --n_layers 3 --epochs 30 --device cpu --item_init outputs/item_llm_emb.npy"

# TwoTower with frozen LLM item embeddings
run_step "08_twotower" \
    "python -u src/two_tower.py --n $N --seed $SEED --tag $TAG --epochs 30 --device cpu"

# =====================================================================
# Phase C: New LLM predictors (need vLLM; run on 72B for quality)
# =====================================================================
switch_to_72b

export CF_LLM_MODEL="$M72_NAME"
export CF_LLM_BASE_URL="http://localhost:8002/v1"

# LLM soft rating distribution (uses retrieval if available)
run_step "09_soft_distribution_72b" \
    "python -u src/run_soft_distribution.py --n $N --concurrency $CONC_72 --seed $SEED --tag $TAG --use_retrieval"

# Preference chain-of-thought
run_step "10_preference_cot_72b" \
    "python -u src/run_preference_cot.py --n $N --concurrency $CONC_72 --seed $SEED --tag $TAG --calibrate"

# Analogical rating via retrieval
run_step "11_analogical_72b" \
    "python -u src/run_analogical.py --n $N --concurrency $CONC_72 --seed $SEED --tag $TAG --k 8 --calibrate"

# LLM confidence router over all available base predictions
run_step "12_confidence_router_72b" \
    "python -u src/run_confidence_router.py --n $N --concurrency $CONC_72 --seed $SEED --tag $TAG"

# =====================================================================
# Phase D: Final stacker + routed analysis
# =====================================================================

run_step "13_final_stacker_v2" \
    "python -u src/stacker.py --tag $TAG --model_suffixes 14b,72b"

run_step "14_routed_hybrid_v2" \
    "python -u src/routed_hybrid.py --tag $TAG"

log "Overnight v2 finished: $(date)"
log "Logs: $LOG_DIR"
log "Final leaderboard in $LOG_DIR/13_final_stacker_v2.log"
