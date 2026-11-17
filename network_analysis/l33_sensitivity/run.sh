#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data=${1:?Supply extracted L33 data directory}
out=${2:?Supply new output directory}
mkdir "$out"
python "$here/ecoli.py" "$data/ecoli" "$out/ecoli.csv" "$data/shap/ecoli.zip"
python "$here/paer_kpn.py" "$data/paer_kpn" "$out/paer_kpn.csv" "$data/shap/archive" "$data/shap/kpn5_corrected.csv"
python "$here/paer_kpn.py" "$data/paer_kpn" "$out/paer_kpn_pilot_shap.csv" "$data/shap/pilot" "$data/shap/kpn5_corrected.csv"
