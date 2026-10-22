# Reproducing the analyses

1. Obtain the data packages listed in `DATA_AND_ZENODO.md`.
2. Install the environment specified by each module.
3. Run the surrogate/SHAP/UMAP and S2 commands in `../post_weka_robustness/REPRODUCING_ANALYSES.md`.
4. Run network selection against its candidate-selection data root using `../network_analysis/README.md`.
5. Run the paired leakage audit using its module README.
6. Generate publication figures using `../publication_figures/README.md`.

Configuration files specify model parameters, selected pools, similarity thresholds and sampling seeds. Task completion records include input fingerprints and output checksums.
