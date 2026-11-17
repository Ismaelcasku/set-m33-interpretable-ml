#!/bin/bash
#SBATCH --partition=cpu20
#SBATCH --nodelist=gorina23
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
src=${1:?RF audit directory}
jar=${2:?WEKA jar}
out="results_rf_${SLURM_JOB_ID}"
mkdir "$out"
sha256sum run.py run.slurm FrozenRF.java inputs/* "$jar" > "$out/sha256.txt"
for model in batch4pseudo10 batch5pseudo20 batch6pseudo10 batch4kleb10 batch6kleb60; do
    find "$src" -type f -name "$model.model" -exec sha256sum {} + >> "$out/sha256.txt"
done
{ python3 -VV; java -version; } > "$out/environment.txt" 2>&1
python3 run.py "$out" "$src" "$jar"
