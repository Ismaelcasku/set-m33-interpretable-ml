#!/usr/bin/env bash
#SBATCH --job-name=weka_leakage_audit
#SBATCH --partition=cpu20
#SBATCH --array=0-549%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/leakage_fold_%A_%a.out
#SBATCH --error=logs/leakage_fold_%A_%a.err

set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
cd "$project_root"
python_bin="${PROJECT_PYTHON:-$project_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found: $python_bin" >&2
  exit 1
fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAVA_HEAP="${JAVA_HEAP:-12g}"
task_offset="${AUDIT_TASK_OFFSET:-0}"
task_id="$((SLURM_ARRAY_TASK_ID + task_offset))"

"$python_bin" scripts/run_leakage_fold.py \
  --project-root "$project_root" \
  --task-id "$task_id"
