#!/bin/bash
#SBATCH --partition=cpu20
#SBATCH --nodelist=gorina23
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
jar=${1:?WEKA jar}
out="results_kpn5_compare_${SLURM_JOB_ID}"
mkdir "$out"
sha256sum run.py run.slurm HistoricalRF.java FrozenRF.java compare.py compare.slurm inputs/* "$jar" > "$out/sha256.txt"
python3 run.py "$out" . "$jar"
python3 compare.py "$out" "$jar"
