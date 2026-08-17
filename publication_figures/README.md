# SET-M33 publication figures v2.1

Generated from the completed post-WEKA robustness results. The default local
input directory is:

`/home/ismael/Desktop/shap_umap/SET_M33_post_WEKA_results_review_v2.1`

The figures use the archived reference SHAP importances and the precomputed
historical grouped Euclidean UMAP coordinates. No model or embedding is refit.

## Contents

- `figures/shap_top20/`: individual charge panels and species composites.
- `figures/shap_heatmaps/`: individual species panels and the Figure 2 composite.
- `figures/historical_umap/`: individual species panels and the Figure 3 composite.
- `source_data/`: exact plotting tables used for every figure.
- `figure_manifest.csv`: figure-to-source mapping.
- `CAPTIONS.md`: draft manuscript captions with the corrected UMAP method.

Every figure is exported as high-resolution PNG, PDF and SVG. Individual PNG
panels use 600 dpi; multi-panel PNGs are capped at 4,800 pixels on the longest
edge because the PDF/SVG versions already preserve print-quality line work.
UMAP point layers are rasterized inside PDF/SVG to keep file sizes manageable;
axes and text remain vector elements.
