#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SET_M33_RESULTS_ROOT="${SET_M33_SMOKE_RESULTS_ROOT:-results_smoke}"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$ROOT/.cache/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
python scripts/preflight.py
python scripts/run_surrogate_cv.py --task-index 0 --repeats 2 --splits 2 --candidate-bootstrap-replicates 20
for TASK_INDEX in 0 1 2 3; do
  python scripts/run_shap_reference.py --task-index "$TASK_INDEX" --evaluation-max-rows 100
done
python scripts/run_shap_seed.py --task-index 0 --seed-index 0 --evaluation-max-rows 100
python scripts/run_shap_bootstrap.py --task-index 0 --chunk-index 0 --total-replicates 1 --chunk-size 1 --evaluation-max-rows 100
python scripts/run_hspn_stability.py --task-index 0 --louvain-seeds 3
python scripts/run_historical_umap.py --species-index 0
python scripts/run_umap_robustness.py --umap-task-index 0 --seeds 42,1
python scripts/aggregate_results.py --allow-partial
echo "Smoke test completed in $SET_M33_RESULTS_ROOT"
