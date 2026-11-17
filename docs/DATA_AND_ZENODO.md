# Data locations

GitHub contains analysis code, configuration, environment files, tests and publication-figure material. Zenodo contains study data, fitted models and numerical outputs.

Study-data archive: https://doi.org/10.5281/zenodo.24281130

| Package | Contents |
|---|---|
| 01 | Training data, descriptors and WEKA models |
| 02 | Generated candidate libraries and WEKA predictions |
| 03 | Experimental MIC measurements |
| 04 | Leakage-controlled validation results |
| 05 | Surrogate, SHAP, UMAP, HSPN robustness and S2 outputs |
| 06 | Similarity networks and candidate-selection outputs |

## Interpretability inputs

Use the complete candidate predictions in package 02 and attribution/projection outputs in package 05. Candidate counts are E. coli 3,918; P. aeruginosa 5,383; K. pneumoniae 1,174. E. coli +5 contains 1,874 candidates.

The preparation scripts map the package directories to the module input layout. Commands are in `../post_weka_robustness/REPRODUCING_ANALYSES.md`.

## Candidate-selection inputs

Network selection and HSPN use package 02 at https://doi.org/10.5281/zenodo.22014734 in a separate data root. Input filenames and thresholds are listed in the network task registry. Network outputs are in package 06.

## Leakage validation

Place the eleven full-descriptor CSVs in `leakage_controlled_validation/input/`, using the names in `config/project.json`. Provide WEKA 3.8.6 as `software/weka.jar` or through `WEKA_JAR`.

## Supplementary Tables S6–S8

The companion data package `SET_M33_S6_S8_data.tar.gz` contains archived inputs, reference CSV outputs and SHA-256 checksums. After extracting it, run the following from the repository root, using Python 3.9 or newer. Replace the path with the actual extraction path.

```bash
DATA=/path/to/SET_M33_S6_S8_data
(cd "$DATA" && sha256sum -c SHA256SUMS)
python3 network_analysis/l33_sensitivity/summarize_table_s6.py \
  "$DATA/inputs/all_modules_l33_491421.csv" table_s6.csv
python3 scaffold_transfer/summarize_predictions.py \
  "$DATA/inputs/SET_M33_S8_predictions.tar.gz" \
  "$DATA/inputs/tetramers_MIC_no_exact_training_overlap.csv" reproduced_s7_s8 \
  --coverage  "$DATA/inputs/ecoli_arm_projection_490408.csv" \
  "$DATA/inputs/paer_kpn_arm_projection_490534.csv"
cmp table_s6.csv "$DATA/results/table_s6.csv"
cmp reproduced_s7_s8/table_s7.csv "$DATA/results/table_s7.csv"
cmp reproduced_s7_s8/table_s8.csv "$DATA/results/table_s8.csv"
```

This reproduces the tables from archived outputs. The upstream Gorina launchers and their staging requirements are documented in [coverage replay](../network_analysis/coverage_replay/README.md) and [WEKA replay](../scaffold_transfer/weka_replay/README.md).
