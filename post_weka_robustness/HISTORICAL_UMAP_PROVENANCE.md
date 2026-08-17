# Historical UMAP provenance

The recovered `UMAP_shap` executable notebook contains the following sequence:

1. concatenate charge-specific SHAP tables;
2. calculate mean signed SHAP by charge and transpose to descriptors × charges;
3. fill missing descriptor/charge values with zero;
4. apply `StandardScaler` to that descriptor-dynamics matrix;
5. apply `KMeans(..., random_state=42)`;
6. calculate each candidate's mean SHAP within each descriptor group using
   pandas row-mean semantics, then fill completely absent groups with zero; and
7. apply `UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
   random_state=42)` without specifying `metric`, hence Euclidean.

The executed code shows `k=6` in the SHAP workflow. Its input paths refer to
*P. aeruginosa* while one output filename refers to *E. coli*, consistent with a
copied exploratory notebook. The related recovered *K. pneumoniae* notebook
records `k=3`. The selected reconstruction therefore uses `k=6`, `k=6` and
`k=3` for *E. coli*, *P. aeruginosa* and *K. pneumoniae*, respectively, as
agreed for the historical reproduction.

The old code did not specify `n_init`; the reconstructed package fixes
`n_init=10`, matching the scikit-learn default used by the historical execution
and preventing version-dependent changes.

The manuscript prose describing direct cosine UMAP conflicts with this
executable evidence. This package follows the recovered code and the Methods
text generated after aggregation describes the grouped Euclidean workflow.
