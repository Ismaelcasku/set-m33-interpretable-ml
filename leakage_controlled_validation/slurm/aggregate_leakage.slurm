#!/usr/bin/env bash
#SBATCH --job-name=weka_leakage_summary
#SBATCH --partition=cpu20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/leakage_aggregate_%j.out
#SBATCH --error=logs/leakage_aggregate_%j.err

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
"$python_bin" scripts/aggregate_leakage_audit.py --project-root "$project_root"
