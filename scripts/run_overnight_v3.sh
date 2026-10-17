#!/bin/bash
# Overnight v3: close the oracle gap by adding:
#   1. Multi-seed NeuMF ensemble (8 configs on GPU)
#   2. DeepFM with side features (genres, demographics, genre affinity)
#   3. Residual NeuMF (gradient boosting with neural base learners)
#   4. Enhanced stacker with LightGBM + rich meta-features
#
# Uses GPU 0 for neural model training (GPU 1 reserved for vLLM).
# All steps logged under outputs/overnight_v3_<timestamp>/.
set -u

cd "$(dirname "$0")/.."   # run from the repository root
# Activate your Python environment (e.g. `conda activate cf`) before running this script.

export CUDA_VISIBLE_DEVICES=0

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=outputs/overnight_v3_${STAMP}
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/master.log"

TAG=n2000
SEED=0
N=2000

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER"; }

run_step() {
    local step_name=$1
    shift
    log "=== START: $step_name ==="
    log "  cmd: $*"
    "$@" 2>&1 | tee -a "$LOG_DIR/${step_name}.log" | tail -20 | tee -a "$MASTER"
    local rc=${PIPESTATUS[0]}
    log "=== END:   $step_name (exit $rc) ==="
    echo "" >> "$MASTER"
    return $rc
}

log "Overnight v3 started: $(date)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0 2>/dev/null || echo 'CPU only')"
log "Logs: $LOG_DIR"
log ""

# Step 1: Multi-seed NeuMF ensemble on GPU
run_step "01_neumf_ensemble" \
    python -u src/neumf_ensemble.py --n $N --seed $SEED --tag $TAG --epochs 40 --device cuda

# Step 2: DeepFM with side features
run_step "02_deepfm" \
    python -u src/deepfm.py --n $N --seed $SEED --tag $TAG --epochs 50 --dim 64 --device cuda

# Step 3: Residual NeuMF
run_step "03_residual_neumf" \
    python -u src/residual_neumf.py --n $N --seed $SEED --tag $TAG --epochs 40 --residual_epochs 30 --device cuda

# Step 4: Re-run v2 stacker to ensure baseline numbers
run_step "04_stacker_v2_baseline" \
    python -u src/stacker.py --tag $TAG --model_suffixes 14b,72b

# Step 5: Enhanced stacker v3 with all new features + LightGBM
run_step "05_stacker_v3" \
    python -u src/stacker_v3.py --tag $TAG --model_suffixes 14b,72b

log ""
log "Overnight v3 finished: $(date)"
log "Logs: $LOG_DIR"
log "Final leaderboard in $LOG_DIR/05_stacker_v3.log"
