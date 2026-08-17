#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results
source "$ROOT/activate_env.sh"
export MPLBACKEND=Agg
TASK_CACHE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}"
TASK_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/setm33_cache_${TASK_CACHE_ID}"
export MPLCONFIGDIR="$TASK_CACHE_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$TASK_CACHE_ROOT/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
unset TASK_CACHE_ID TASK_CACHE_ROOT
