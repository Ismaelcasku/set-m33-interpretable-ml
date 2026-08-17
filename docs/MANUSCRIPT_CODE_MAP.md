# Manuscript-to-code map

| Manuscript analysis / output | Canonical repository location | Main entry points |
|---|---|---|
| Physicochemical similarity-network construction | `network_analysis/` | `scripts/run_network_analysis.py`, `config/tasks.csv` |
| Candidate selection from modules/PageRank/predicted MIC | `network_analysis/` | `scripts/run_network_analysis.py::select_candidates` |
| Figure 1 graph topology / Gephi export | `network_analysis/`, `docs/FIGURE_1_GEPHI.md` | `nodes_selected_threshold.csv`, `edges_selected_threshold.csv` |
| Leakage-controlled feature selection / RF evaluation | `leakage_controlled_validation/` | `scripts/preflight_leakage_audit.py`, `scripts/run_global_selection.py`, `scripts/run_leakage_fold.py`, `scripts/aggregate_leakage_audit.py` |
| `legacy_global` vs `nested_matched` comparison | `leakage_controlled_validation/` | `config/leakage_audit.json`, `scripts/submit_leakage_audit.sh` |
| Surrogate fidelity to archived WEKA predictions | `post_weka_robustness/` | `scripts/run_surrogate_cv.py`, `scripts/run_shap_reference.py` |
| Reference TreeSHAP attributions | `post_weka_robustness/` | `scripts/run_shap_reference.py` |
| SHAP RF-seed stability | `post_weka_robustness/` | `scripts/run_shap_seed.py` |
| SHAP bootstrap stability | `post_weka_robustness/` | `scripts/run_shap_bootstrap.py` |
| HSPN threshold/Louvain sensitivity | `post_weka_robustness/` | `scripts/run_hspn_stability.py` |
| Historical grouped SHAP → Euclidean UMAP | `post_weka_robustness/` | `scripts/run_historical_umap.py`, `HISTORICAL_UMAP_PROVENANCE.md` |
| UMAP robustness | `post_weka_robustness/` | `scripts/run_umap_robustness.py` |
| Final post-WEKA summary tables | `post_weka_robustness/` | `scripts/aggregate_results.py` |
| Figure 2 SHAP heatmaps | `publication_figures/` | `generate_historical_figures.py`, `source_data/shap_heatmaps/` |
| Figure 3 historical grouped Euclidean UMAP | `publication_figures/` | `generate_historical_figures.py`, `source_data/historical_umap/` |
| Supplementary SHAP top-20 panels | `publication_figures/` | `generate_historical_figures.py`, `source_data/shap_top20/` |

## Interpretation boundary

The leakage audit evaluates predictive-validation methodology. SHAP values
explain Python surrogate reproductions of archived WEKA predictions. Similarity
networks, Louvain communities, UMAP neighborhoods and HDBSCAN clusters are
algorithmic representations and are not direct evidence of causal membrane
mechanisms or biological classes.
