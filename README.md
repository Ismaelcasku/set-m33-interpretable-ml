# SET-M33 interpretable machine-learning analyses

Code for “Interpretable machine learning identifies target-dependent predictive patterns for a dendrimeric antimicrobial peptide.”

## Contents

- `network_analysis/`: physicochemical similarity networks and candidate selection.
- `leakage_controlled_validation/`: paired global and nested feature-selection validation.
- `post_weka_robustness/`: surrogate fidelity, SHAP, grouped-SHAP UMAP and robustness analyses.
- `post_weka_robustness/s2_tables/`: effective dimensionality and charge-dependent PCA/UMAP metrics for Tables S2.1–S2.3.
- `publication_figures/`: figure generator, plotting data and rendered figures.

## Data

Study data and numerical outputs: https://doi.org/10.5281/zenodo.20530642

Interpretability includes 3,918 E. coli analogs (1,874 at charge +5), 5,383 P. aeruginosa analogs and 1,174 K. pneumoniae analogs. Candidate-selection inputs are available in package 02 at https://doi.org/10.5281/zenodo.22014734. These two analyses use separate data roots.

See `docs/DATA_AND_ZENODO.md` for file locations and `post_weka_robustness/REPRODUCING_ANALYSES.md` for execution commands.

## Methods

Random Forest surrogates are fitted to WEKA predictions. TreeSHAP provides descriptor-level attributions. Descriptors are grouped from their charge-dependent mean signed SHAP profiles using zero filling, StandardScaler and KMeans (six groups for E. coli and P. aeruginosa; three for K. pneumoniae). Each candidate is represented by its within-group mean signed SHAP and projected by Euclidean UMAP.

UMAP geometry metrics use the complete candidate library. Parameter-grid charge metrics use a fixed balanced subset. Tables S2.1–S2.3 use 20 balanced draws per charge; their calculation is specified in `post_weka_robustness/s2_tables/METHODS.md`.

The leakage audit compares `legacy_global` and `nested_matched` on 11 datasets, ten seeds and five folds: 1,100 outer Random Forest fits. Network construction, thresholds and candidate ranking are described in `network_analysis/README.md`.

## Citation and license

Software metadata are provided in `CITATION.cff`. Cite the associated manuscript and study-data archive. No explicit software license is assigned.
