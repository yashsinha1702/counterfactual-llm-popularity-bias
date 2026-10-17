#!/bin/bash
# Overnight dual-model sequential run.
# Sequence:
#   1. Embed movies
#   2. Launch NeuMF training on CPU in background (parallel)
#   3. 14B is already serving on port 8002 — run retrieval-diverse + retrieval-CFv2 with tag n2000_14b
#   4. Kill 14B vLLM, launch 72B vLLM on GPU 1 port 8002
#   5. Run retrieval-diverse + retrieval-CFv2 with tag n2000_72b
#   6. Wait for NeuMF to finish
#   7. Final stacker merging everything (both models + NeuMF + classical CF)

set -u

cd "$(dirname "$0")/.."   # run from the repository root
# Activate your Python environment (e.g. `conda activate cf`) before running this script.

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=outputs/overnight_${STAMP}
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/master.log"

VLLM_BIN=${VLLM_BIN:-vllm}
M72_PATH=${M72_PATH:-models/qwen2.5-72b-instruct-awq}   # local AWQ weights (or a HF repo id)
M72_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
M14_NAME=Qwen/Qwen2.5-14B-Instruct

log()    { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
step()   { echo -e "\n[$(date +%H:%M:%S)] === $* ===" | tee -a "$MASTER"; }

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
        tail -6 "$step_log" | tee -a "$MASTER"
    fi
    return $rc
}

wait_for_ready() {
    # Polls the model list endpoint until vLLM responds, or timeout
    local timeout=${1:-600}
    local waited=0
    while [ $waited -lt $timeout ]; do
        if curl -sf http://localhost:8002/v1/models > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

switch_vllm_to_72b() {
    step "Switching vLLM: 14B -> 72B on GPU 1"
    local pid=$(pgrep -f "vllm serve" | head -1)
    if [ -n "$pid" ]; then
        log "  killing vLLM PID $pid ..."
        kill "$pid"
        sleep 10
        if kill -0 "$pid" 2>/dev/null; then
            log "  not exiting gracefully, SIGKILL"
            kill -9 "$pid"
            sleep 3
        fi
    fi
    # Wait for GPU to free
    sleep 20
    log "  launching 72B server ..."
    CUDA_VISIBLE_DEVICES=1 nohup "$VLLM_BIN" serve \
        "$M72_PATH" \
        --served-model-name "$M72_NAME" \
        --host 127.0.0.1 --port 8002 \
        --gpu-memory-utilization 0.90 --max-model-len 8192 \
        > "$LOG_DIR/vllm_72b_server.log" 2>&1 &
    log "  waiting for 72B readiness ..."
    if wait_for_ready 600; then
        log "  72B ready."
        return 0
    else
        log "  72B failed to become ready within 10 min. Aborting 72B phase."
        return 1
    fi
}

log "Overnight dual-model run started: $STAMP"
log "Log directory: $LOG_DIR"

# ----- 1. Embed movies (shared across models) -----
run_step "01_embed_movies" "python -u src/embed_movies.py"

# ----- 2. NeuMF on CPU in background (doesn't need vLLM) -----
step "Launching NeuMF (CPU) in background"
nohup python -u src/neumf.py --n 2000 --seed 0 --tag n2000 --epochs 25 --dim 32 --device cpu \
    > "$LOG_DIR/02_neumf.log" 2>&1 &
NEUMF_PID=$!
log "  NeuMF PID $NEUMF_PID (will sync at end)"

# ----- 3. 14B phase (server already running on GPU 1, serving Qwen2.5-14B-Instruct) -----
export CF_LLM_MODEL="$M14_NAME"
export CF_LLM_BASE_URL="http://localhost:8002/v1"

run_step "03_14b_diverse_retrieval_n2000" \
    "CF_LLM_MODEL=$M14_NAME python -u src/run_diverse_retrieval.py --n 2000 --concurrency 16 --seed 0 --tag n2000_14b"

run_step "04_14b_cfv2_retrieval_n2000" \
    "CF_LLM_MODEL=$M14_NAME python -u src/run_cf_v2_retrieval.py --n 2000 --concurrency 16 --seed 0 --alpha 0.5 --calibrate --tag n2000_14b"

# ----- 4. Switch to 72B -----
if switch_vllm_to_72b; then
    export CF_LLM_MODEL="$M72_NAME"
    run_step "05_72b_diverse_retrieval_n2000" \
        "CF_LLM_MODEL=$M72_NAME python -u src/run_diverse_retrieval.py --n 2000 --concurrency 12 --seed 0 --tag n2000_72b"

    run_step "06_72b_cfv2_retrieval_n2000" \
        "CF_LLM_MODEL=$M72_NAME python -u src/run_cf_v2_retrieval.py --n 2000 --concurrency 12 --seed 0 --alpha 0.5 --calibrate --tag n2000_72b"
else
    log "Skipping 72B phase due to vLLM startup failure."
fi

# ----- 5. Wait for NeuMF (launched in parallel) -----
step "Waiting for NeuMF to finish"
while kill -0 "$NEUMF_PID" 2>/dev/null; do sleep 20; done
log "  NeuMF finished."
tail -10 "$LOG_DIR/02_neumf.log" | tee -a "$MASTER"

# ----- 6. Final stacker: merge everything -----
run_step "07_final_stacker" \
    "python -u src/stacker.py --tag n2000 --model_suffixes 14b,72b"

# ----- 7. Routed hybrid (quick sanity) -----
run_step "08_routed_hybrid" "python -u src/routed_hybrid.py --tag n2000"

log "Overnight run finished: $(date)"
log "All logs: $LOG_DIR"
log "Key numbers in $LOG_DIR/07_final_stacker.log"
