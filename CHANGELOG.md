# Changelog

## 1.0.0 — 2026-08-17

Paper-aligned GitHub code release.

- Added standalone physicochemical-similarity-network and candidate-selection
  module containing the analysis logic used in the study.
- The public release contains only manuscript-relevant code paths; superseded
  exploratory end-to-end notebooks are intentionally excluded.
- Leakage audit fixed to 11 datasets and two branches only:
  `legacy_global` and `nested_matched` (1,100 outer RF fits).
- Canonical K. pneumoniae +5/+6 historical pools fixed at pool60/pool60;
  `nested_current` is not included.
- Post-WEKA historical pools fixed at E. coli 10/10/10/20, P. aeruginosa
  10/20/10/0 and K. pneumoniae 10/60/60.
- Post-WEKA package normalized to version 2.1.0.
- UMAP robustness corrected so trustworthiness, exact kNN preservation and
  between-run neighborhood overlap are calculated on all candidates; charge
  silhouettes and HDBSCAN remain charge-balanced.
- Added the exact publication-figure package for Figure 2, Figure 3 and
  supplementary SHAP top-20 panels, including source-data tables and rendered
  outputs.
- Large study data and numerical HPC outputs remain assigned to Zenodo.
