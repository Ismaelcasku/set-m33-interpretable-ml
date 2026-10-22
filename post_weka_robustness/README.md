# Post-WEKA analysis

The Python Random Forest surrogates model the WEKA `predicted` column. TreeSHAP provides descriptor-level attribution profiles.

## Analyses

- Fidelity: fitted R², MAE, RMSE and Spearman correlation; ten repetitions of five-fold CV; paired candidate bootstrap intervals.
- SHAP stability: 20 Random Forest seeds and 200 bootstrap training samples, with a common evaluation subset of up to 1,000 candidates.
- Grouped-SHAP UMAP: descriptor-by-charge mean signed SHAP, zero filling, StandardScaler, KMeans descriptor groups (6/6/3), candidate-level within-group mean SHAP and unscaled Euclidean UMAP.
- UMAP robustness: neighborhood sizes 5/15/50, minimum distances 0.0/0.1/0.5 and ten seeds. Geometry metrics use the complete candidate library; charge metrics use a fixed balanced subset.
- Tables S2.1–S2.3: full descriptor-level effective dimensionality and charge-dependent PCA/UMAP metrics from 20 balanced draws. See `s2_tables/METHODS.md`.
- HSPN robustness: cosine thresholds and 100 Louvain seeds; modularity, partition agreement, PageRank stability and candidate-selection frequency.

## Execution

Install the environment using `setup_environment.sh` and activate it with `activate_env.sh`. Input paths and parameters are defined in `config/tasks.csv` and `config/robustness.yml`. Input hashes are in `data/input_manifest.sha256`.

See `REPRODUCING_ANALYSES.md` for package-02/05 preparation, the E. coli +5 SHAP/UMAP workflow and S2 execution. HSPN uses the candidate-selection data root specified in `../docs/DATA_AND_ZENODO.md`. The `submit_all.sh` launcher executes all stages, including HSPN.

Outputs include surrogate fidelity, SHAP arrays and importance tables, seed/bootstrap summaries, grouped matrices, primary embeddings and UMAP parameter-grid metrics. Each completed task records input fingerprints and output checksums.
