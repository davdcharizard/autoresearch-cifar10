#!/usr/bin/env bash
# EXP-014 driver: run the three cells as SEPARATE train.py processes, each under its
# own `timeout 600` (no single process breaches the 10-min wall). NO retries, NO
# selection logic — just c0, cA, cB in order. Logs nvidia-smi before each cell.
# (Recorded verbatim in 03-execute.md per plan-review concern #9.)
set -u
cd /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8
EXPDIR=.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014
CACHE=$(pwd)/$EXPDIR/.inductor_cache

run_cell () {
  local name=$1 use=$2 width=$3
  echo "===== CELL ${name}: USE_COMPILE=${use} WARMUP=1 LAYER2_WIDTH=${width} ====="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | sed 's/^/[smi] /'
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$(pwd) TORCHINDUCTOR_CACHE_DIR=$CACHE \
    USE_COMPILE=$use WARMUP=1 LAYER2_WIDTH=$width COMPILE_MODE=default \
    timeout 600 uv run train.py > $EXPDIR/run_${name}.log 2>&1
  echo "cell ${name} exit: $?"
}

run_cell c0 0 256   # no-compile control (with off-budget warmup parity)
run_cell cA 1 256   # compile, 256
run_cell cB 1 320   # compile, 320 (headline)
echo "ALL CELLS DONE"
