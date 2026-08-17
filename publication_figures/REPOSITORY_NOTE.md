# Repository note

This directory is the publication-figure package used after the post-WEKA
robustness analysis. `generate_historical_figures.py` is preserved as supplied.
It does not refit Random Forest surrogates, recalculate SHAP, or rerun UMAP; it
renders figures from the completed post-WEKA result tables/coordinates.

The checked-in `source_data/` directory contains the exact plotting tables
exported by that script, and `figures/` contains the corresponding final
PNG/PDF/SVG outputs for Figure 2, Figure 3, and the supplementary SHAP top-20
panels. The original script has workstation-specific default paths; use explicit
`--results-root` and `--output-root` arguments on another machine.

Figure 1 is not generated here because its final rendering was performed in
Gephi. See `../network_analysis/` and `../docs/FIGURE_1_GEPHI.md`.
