#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent="$(cd "$root/.." && pwd)"
archive="$parent/01_SET_M33_training_data.tar.gz?download=1"
mic=tetramers_MIC_no_exact_training_overlap.csv
[[ -r "$archive" && -r "$parent/$mic" && -d "$parent/data" ]]
ln -s -- "../$mic" "$root/$mic"
ln -s -- ../data "$root/data"
mkdir -p "$root/logs"
mkdir "$root/models"
tar -xzf "$archive" -C "$root/models" -- \
  'Training/WEKA_models/ecoli/batch 4/ecol10_batch4.model' \
  'Training/WEKA_models/ecoli/batch 5/batch5ecol10.model' \
  'Training/WEKA_models/ecoli/batch 6/batch6ecol10.model' \
  'Training/Subsets_pool_descriptors/ecoli/subset4/ecoli_training_pool_10.csv' \
  'Training/Subsets_pool_descriptors/ecoli/subset5/ecoli5_training_pool_10.csv' \
  'Training/Subsets_pool_descriptors/ecoli/subset6/ecoli6_training_pool_10.csv'
