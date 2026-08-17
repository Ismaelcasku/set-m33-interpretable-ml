# Generate the SET-M33 historical figures locally

The script defaults to the directories requested for the local workstation:

```bash
cd /home/ismael/Desktop/shap_umap
python generate_historical_figures.py
```

Explicit invocation:

```bash
python generate_historical_figures.py \
  --results-root /home/ismael/Desktop/shap_umap/SET_M33_post_WEKA_results_review_v2.1 \
  --output-root /home/ismael/Desktop/shap_umap/SET_M33_publication_figures_v2.1
```

Required packages: `numpy`, `pandas`, `matplotlib`, and `seaborn`. UMAP and SHAP
are not required because the script uses the archived coordinates and
importance tables.
