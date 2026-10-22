# Data locations

GitHub contains analysis code, configuration, environment files, tests and publication-figure material. Zenodo contains study data, fitted models and numerical outputs.

Study-data archive: https://doi.org/10.5281/zenodo.20530642

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
