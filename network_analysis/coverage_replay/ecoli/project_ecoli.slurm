#!/bin/bash
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"${PYTHON:-python3}" project_ecoli.py inputs "results_${SLURM_JOB_ID}"
