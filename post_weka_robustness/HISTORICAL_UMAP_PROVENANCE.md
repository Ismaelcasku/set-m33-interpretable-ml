# Grouped-SHAP UMAP

Charge-specific signed SHAP values are summarized by descriptor and charge. Absent descriptor/charge combinations are zero-filled for descriptor clustering. StandardScaler and KMeans identify descriptor groups (6 for E. coli, 6 for P. aeruginosa, 3 for K. pneumoniae; seed 42, n_init 10). Candidate representations are within-group mean signed SHAP values, projected by Euclidean UMAP with 15 neighbors, min_dist 0.1 and seed 42.
