# Publication figures

Current Figures 2/3 and supplementary SHAP panels are in `figures/`. Figure 2 source tables are `figures/*_Figure2_values.csv` and `*_relative_SHAP_all.csv`; Figure 3 coordinates are in `source_data/historical_umap/`. E. coli includes 3,918 candidates; its +5 surrogate includes 1,874. 

The generator renders the SHAP arrays and primary projection coordinates as PNG, SVG and PDF files.

```bash
python publication_figures/generate_historical_figures.py \
  --results-root /path/to/Post_WEKA_robustness \
  --output-root /path/to/new_figures
```

Use the post-WEKA pinned environment.
