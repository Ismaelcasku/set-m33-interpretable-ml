# Draft figure captions

## Figure 2. SHAP feature importance across charge states

Heatmaps show the ten descriptors with the highest mean relative absolute SHAP
importance across charge states for each bacterial target: (A) *E. coli* TG1,
(B) *P. aeruginosa* PAO-1, and (C) *K. pneumoniae* ATCC 13833. Within each
charge-specific surrogate, mean absolute SHAP values were divided by their sum
over all descriptors; absent descriptor/charge combinations are displayed as
zero. SHAP importance quantifies the contribution of a descriptor to the
Python Random Forest surrogate of the archived WEKA `predicted` values and
does not establish biological causality or directionality.

## Figure 3. Historical grouped Euclidean UMAP

Historical UMAP projection of grouped SHAP values for each bacterial target:
(A) *E. coli* TG1, (B) *P. aeruginosa* PAO-1, and (C) *K. pneumoniae* ATCC
13833. Each point represents one candidate analog and is colored by cationic
charge. Descriptors were grouped from their charge-dependent mean signed SHAP
dynamics by KMeans (six groups for *E. coli*, six for *P. aeruginosa*, and
three for *K. pneumoniae*), and each analog was represented by the mean SHAP
value within each group. The grouped matrix was projected without additional
scaling by Euclidean UMAP (`n_neighbors=15`, `min_dist=0.1`, two components,
`random_state=42`) without subsampling. Axis units are arbitrary; cluster
geometry is shown descriptively and is not interpreted as evidence of
biological classes.

## Supplementary SHAP top-20 panels

The twenty descriptors with the highest relative mean absolute SHAP value are
shown for each target- and charge-specific surrogate. Values were normalized
to sum to one within each model to permit comparison of relative importance
across libraries. These plots describe the surrogate fitted to archived WEKA
predictions rather than the serialized WEKA trees.
