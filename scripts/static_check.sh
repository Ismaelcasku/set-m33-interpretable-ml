#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

python -m py_compile \
  network_analysis/scripts/run_network_analysis.py \
  leakage_controlled_validation/scripts/*.py \
  post_weka_robustness/scripts/*.py \
  post_weka_robustness/s2_tables/*.py \
  post_weka_robustness/setm33robust/*.py \
  publication_figures/generate_historical_figures.py

while IFS= read -r -d '' file; do
  bash -n "$file"
done < <(find leakage_controlled_validation post_weka_robustness -type f \
  \( -name '*.sh' -o -name '*.slurm' -o -name '*.sbatch' \) -print0)

echo "Static checks passed."
