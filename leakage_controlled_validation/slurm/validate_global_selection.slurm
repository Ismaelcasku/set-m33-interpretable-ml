#!/usr/bin/env bash
#SBATCH --job-name=weka_validate_pools
#SBATCH --partition=cpu20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/leakage_validate_pools_%j.out
#SBATCH --error=logs/leakage_validate_pools_%j.err

set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
cd "$project_root"
python_bin="${PROJECT_PYTHON:-$project_root/.venv/bin/python}"
"$python_bin" scripts/validate_global_selection.py --project-root "$project_root"
