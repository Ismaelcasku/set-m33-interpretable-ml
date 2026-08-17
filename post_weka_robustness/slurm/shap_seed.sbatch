#!/usr/bin/env bash
#SBATCH --job-name=setm33_seed
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"
N_SEEDS="$(python scripts/workload.py shap_seed_count)"
TASK_INDEX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
python scripts/run_shap_seed.py --task-index "$TASK_INDEX" --seed-index "$SEED_INDEX"
