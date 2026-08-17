# Reproducibility hierarchy

Use the following hierarchy when reproducing the manuscript:

1. **Canonical analyses**
   - `network_analysis/` for physicochemical similarity networks and candidate prioritization
   - `leakage_controlled_validation/` for the paired leakage audit
   - `post_weka_robustness/` for surrogate fidelity, SHAP robustness, HSPN sensitivity, historical grouped-SHAP UMAP and UMAP robustness
2. **Publication rendering**
   - `publication_figures/` for Figures 2/3 and supplementary SHAP plots
   - `docs/FIGURE_1_GEPHI.md` for the final Figure 1 rendering step in Gephi

The repository intentionally contains only the code paths needed to reproduce the analyses reported in the revised manuscript. Earlier exploratory or superseded end-to-end workflows are not distributed here.

The final UMAP robustness implementation evaluates geometry fidelity on all candidates while reserving the fixed charge-balanced subset for charge-dependent silhouettes and HDBSCAN stability.
