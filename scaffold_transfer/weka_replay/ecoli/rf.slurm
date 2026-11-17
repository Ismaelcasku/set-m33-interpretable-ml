#!/usr/bin/env bash
#SBATCH --job-name=m33_tet_rf
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
set -euo pipefail
cd -- "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash run_rf.sh "$@"
