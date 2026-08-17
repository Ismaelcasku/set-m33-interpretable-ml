# GitHub / Zenodo data boundary

## GitHub

GitHub is the human-readable software repository. It contains:

- analysis source code;
- canonical configuration and task registries;
- unit tests and environment files;
- lightweight input hash manifests;
- exact plotting source-data tables for Figures 2 and 3;
- final rendered publication figures from the supplied figure package.

## Zenodo

Zenodo is the archival source for the large study data and immutable numerical
outputs. The current study-data DOI is `10.5281/zenodo.20610591`.

The final manuscript-aligned Zenodo version should retain the existing training,
test, similarity-network and MIC data and add the final outputs from:

1. the paired leakage audit (`legacy_global` and `nested_matched`), including
   input/manifests, aggregate tables, OOF predictions and feature-selection
   stability outputs;
2. surrogate full-fit and repeated-CV fidelity;
3. reference SHAP, 20 RF-seed runs and 200 bootstrap SHAP runs;
4. HSPN threshold/Louvain robustness outputs;
5. historical grouped-SHAP/Euclidean UMAP reconstruction;
6. corrected UMAP robustness with all-candidate geometry metrics and
   charge-balanced charge-dependent metrics;
7. final summary tables used to update Results/Supporting Information.

The Zenodo README should point to the GitHub repository for the maintained code.
At code/data freeze, archive or link the exact GitHub `v1.0.0` release so the
published numerical outputs and source code remain paired.

## Expected data layouts

### Network analysis

`network_analysis/` expects an extracted `Supplementary Data` root containing
both `Training/` and `Test/`; paths are defined in `network_analysis/config/tasks.csv`.

### Leakage audit

Place the 11 full-descriptor CSVs in:

```text
leakage_controlled_validation/input/
```

using the exact names in `leakage_controlled_validation/config/project.json`.
WEKA 3.8.6 is expected separately at `software/weka.jar` or via `WEKA_JAR`.

### Post-WEKA analysis

Set `SET_M33_DATA_ROOT` to the extracted `Supplementary Data` directory. The
11 candidate-prediction filenames, selected pools and thresholds are fixed in
`post_weka_robustness/config/tasks.csv`. Expected input SHA-256 hashes are stored
in `post_weka_robustness/data/input_manifest.sha256`.
