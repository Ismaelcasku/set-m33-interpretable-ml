#!/usr/bin/env bash
#SBATCH --job-name=setm33_boot
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"
N_CHUNKS="$(python scripts/workload.py bootstrap_chunk_count)"
TASK_INDEX=$((SLURM_ARRAY_TASK_ID / N_CHUNKS))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % N_CHUNKS))
python scripts/run_shap_bootstrap.py --task-index "$TASK_INDEX" --chunk-index "$CHUNK_INDEX"
