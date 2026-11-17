#!/bin/bash
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=${PYTHON:-python3}
src=${1:?Pass the historical SHAP directory}
out="results_${SLURM_JOB_ID}"
mkdir -p "$out/shap_inputs"
for s in paeruginosa kpneumoniae; do
    for q in 4 5 6; do
        d="${s}_${q}plus"
        mkdir "$out/shap_inputs/$d"
        cp "$src/$d/reference_importance.csv" "$out/shap_inputs/$d/"
    done
done
"$py" -VV > "$out/python.txt"
"$py" -c "import numpy, pandas, sklearn; print(numpy.__version__, pandas.__version__, sklearn.__version__)" > "$out/environment.txt"
sha256sum run.py run.slurm inputs/* "$out"/shap_inputs/*/*.csv > "$out/sha256.txt"
"$py" run.py inputs "$out" "$out/shap_inputs"
