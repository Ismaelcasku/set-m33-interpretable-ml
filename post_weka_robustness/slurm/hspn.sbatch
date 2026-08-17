#!/usr/bin/env bash
#SBATCH --job-name=setm33_hspn
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"
python scripts/run_hspn_stability.py --task-index "$SLURM_ARRAY_TASK_ID"
