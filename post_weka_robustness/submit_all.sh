#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SUBMISSION_DIR="${SET_M33_SUBMISSION_DIR:-$ROOT/results/submission}"
mkdir -p logs "$SUBMISSION_DIR"
source "$ROOT/activate_env.sh"
python scripts/preflight.py

N_TASKS="$(python scripts/workload.py task_count)"
N_SPECIES="$(python scripts/workload.py species_count)"
N_SEEDS="$(python scripts/workload.py shap_seed_count)"
N_CHUNKS="$(python scripts/workload.py bootstrap_chunk_count)"
N_UMAP="$(python scripts/workload.py umap_task_count)"
TASK_LAST=$((N_TASKS - 1))
SPECIES_LAST=$((N_SPECIES - 1))
SEED_LAST=$((N_TASKS * N_SEEDS - 1))
BOOT_LAST=$((N_TASKS * N_CHUNKS - 1))
UMAP_LAST=$((N_UMAP - 1))

# Every array task requests exactly one CPU. Arrays are chained, so this is a
# genuine global cap for this project. With seven one-CPU WEKA tasks, 13+7=20.
THROTTLE="${SET_M33_THROTTLE:-13}"
if (( THROTTLE < 1 || THROTTLE > 13 )); then
  echo "SET_M33_THROTTLE must be between 1 and 13." >&2
  exit 2
fi

CV_JOB="$(sbatch --parsable --array="0-${TASK_LAST}%${THROTTLE}" slurm/surrogate_cv.sbatch)"
CV_DEP="${CV_JOB%%;*}"
REF_JOB="$(sbatch --parsable --dependency="afterok:${CV_DEP}" --array="0-${TASK_LAST}%${THROTTLE}" slurm/shap_reference.sbatch)"
REF_DEP="${REF_JOB%%;*}"
SEED_JOB="$(sbatch --parsable --dependency="afterok:${REF_DEP}" --array="0-${SEED_LAST}%${THROTTLE}" slurm/shap_seed.sbatch)"
SEED_DEP="${SEED_JOB%%;*}"
BOOT_JOB="$(sbatch --parsable --dependency="afterok:${SEED_DEP}" --array="0-${BOOT_LAST}%${THROTTLE}" slurm/shap_bootstrap.sbatch)"
BOOT_DEP="${BOOT_JOB%%;*}"
HSPN_JOB="$(sbatch --parsable --dependency="afterok:${BOOT_DEP}" --array="0-${TASK_LAST}%${THROTTLE}" slurm/hspn.sbatch)"
HSPN_DEP="${HSPN_JOB%%;*}"
HIST_JOB="$(sbatch --parsable --dependency="afterok:${HSPN_DEP}" --array="0-${SPECIES_LAST}%${THROTTLE}" slurm/historical_umap.sbatch)"
HIST_DEP="${HIST_JOB%%;*}"
UMAP_JOB="$(sbatch --parsable --dependency="afterok:${HIST_DEP}" --array="0-${UMAP_LAST}%${THROTTLE}" slurm/umap.sbatch)"
UMAP_DEP="${UMAP_JOB%%;*}"
AGG_JOB="$(sbatch --parsable --dependency="afterok:${UMAP_DEP}" slurm/aggregate.sbatch)"

{
  printf 'stage\tjob_id\n'
  printf 'surrogate_cv\t%s\n' "$CV_JOB"
  printf 'shap_reference\t%s\n' "$REF_JOB"
  printf 'shap_seeds\t%s\n' "$SEED_JOB"
  printf 'shap_bootstrap\t%s\n' "$BOOT_JOB"
  printf 'hspn\t%s\n' "$HSPN_JOB"
  printf 'historical_umap\t%s\n' "$HIST_JOB"
  printf 'umap_robustness\t%s\n' "$UMAP_JOB"
  printf 'aggregate\t%s\n' "$AGG_JOB"
} > "$SUBMISSION_DIR/job_ids.tsv"
cat "$SUBMISSION_DIR/job_ids.tsv"
echo "Submitted with a global project cap of ${THROTTLE} one-CPU tasks."
echo "Monitor with: bash status.sh"
