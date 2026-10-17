#!/bin/bash
# Run the full experiment suite on the current LLM_MODEL (pointed at the 14B server).
# Reuses cached abduction/intervention outputs if any — 14B cache is separate from 72B.
set -e
cd "$(dirname "$0")/.."   # run from the repository root
# Activate your Python environment (e.g. `conda activate cf`) before running this script.

N=${N:-500}
CONC=${CONC:-16}
SEED=${SEED:-0}
TAG=${TAG:-14b_n${N}}

LOGDIR=outputs/suite_${TAG}_$(date +%s)
mkdir -p "$LOGDIR"
echo "Log dir: $LOGDIR"

echo "[1/3] main pipeline (baseline + CF alpha=0.5) ..."
python -u src/run.py --n "$N" --concurrency "$CONC" --seed "$SEED" --tag "$TAG" \
    > "$LOGDIR/run.log" 2>&1

echo "[2/3] diverse baseline ..."
python -u src/run_diverse.py --n "$N" --concurrency "$CONC" --seed "$SEED" --tag "$TAG" \
    > "$LOGDIR/run_diverse.log" 2>&1

echo "[3/3] TIPS alpha ablation ..."
python -u src/ablate.py --n "$N" --concurrency "$CONC" --seed "$SEED" \
    --alphas 0.0,0.5,1.0 --tag "$TAG" \
    > "$LOGDIR/ablate.log" 2>&1

echo "ALL DONE — logs in $LOGDIR"
ls -la outputs/ | tail -10
