# Methods for Tables S2.1-S2.3

The analysis includes 3,918 E. coli candidates, 5,383 P. aeruginosa candidates and 1,174 K. pneumoniae candidates.

## Representations

Effective dimensionality is calculated independently for each charge from all available full descriptor-level signed SHAP profiles, without feature scaling or grouping. It is the participation ratio of covariance eigenvalues, (sum lambda)^2 / sum(lambda^2). A zero covariance gives NA. Centered singular values give the same ratio without the covariance normalization factor.

PCA uses the current per-candidate mean signed SHAP within the KMeans descriptor groups (six for E. coli and P. aeruginosa, three for K. pneumoniae). It is fitted once jointly to all candidates within each organism, with two components, full SVD, no feature scaling and no whitening. The primary UMAP coordinates used for the current manuscript figures are reused unchanged; those were fitted to the same unscaled grouped SHAP representation with Euclidean distance, 15 neighbors, min_dist 0.1 and seed 42. Effective dimensionality describes descriptor-level SHAP; projection metrics describe grouped SHAP.

## Repeated charge-balanced sampling

For each organism, 20 independent subsampling draws select n_sub candidates without replacement from each charge, where n_sub is the smallest charge library: 127 (E. coli), 38 (P. aeruginosa), 165 (K. pneumoniae). Sampling uses NumPy SeedSequence([42, species_index, repeat_index, charge]), with species order E. coli, P. aeruginosa, K. pneumoniae. Row indices are sorted before clustering; the same selected candidates are used for PCA and UMAP. Sampling is repeated from fixed complete-library projections, not followed by refitting either projection. For the smallest charge, every draw includes the entire charge library, so identical metrics across draws are expected. These draws measure sampling sensitivity, not independent biological replication, and their SD is descriptive.

## Clustering and averaging

HDBSCAN is fitted separately within each charge, space and draw: Euclidean metric, min_cluster_size=5, min_samples=5, cluster_selection_method=eom, allow_single_cluster=False, approx_min_span_tree=False, core_dist_n_jobs=1. Noise has label -1.

- k is the number of non-noise clusters. Its mean includes every draw, including zero-cluster draws; therefore a mean below one is possible.
- Noise is the fraction of all sampled points assigned label -1. Its mean includes every draw, including all-noise draws.
- Intra-r is the arithmetic mean Euclidean distance of assigned points to their own cluster centroid. Thus clusters are weighted by assigned point counts. Noise is excluded. A one-cluster result has a valid radius; an all-noise result has NA radius and is excluded only from the radius mean.
- Silhouette uses Euclidean distance after noise points are removed. It is calculated only when 2 <= number of non-noise clusters < number of assigned points. Otherwise it is NA, not zero, and excluded from the silhouette mean and SD. No error is silently converted to NA.
- A draw with more than 50% noise is flagged but is not automatically discarded if silhouette remains mathematically defined. Such a silhouette describes only the assigned minority; it must be interpreted with the noise fraction and assigned-point coverage. 
- All metric means and sample SDs (ddof=1) use equal weights across valid draws. A mean with no valid draws is NA; an SD with fewer than two valid draws is NA. Valid-draw counts are recorded separately for each metric. Silhouette valid counts are included in the Word tables. Majority-noise counts, zero/one-cluster counts and minimum assigned fractions are in the full-precision tables.

Radii are in arbitrary coordinate units; PCA and UMAP radii are not directly comparable to each other or to values obtained with different preprocessing or separately fitted embeddings. Silhouette and clustering summarize projection geometry, not validated biological classes.

## Suggested shared table note (verify against completed outputs)

n denotes the complete candidate count per charge; n sub is the common charge-balanced sample size. Effective dimensionality is the participation ratio of the covariance of full, unscaled descriptor-level SHAP profiles within each charge. PCA and UMAP metrics refer to the grouped SHAP representations used in the current analysis. Both projections were fitted to the complete library of each organism; clustering was evaluated separately by charge over 20 repeated balanced subsamples. k and noise were averaged over all draws. Intra-r is the point-weighted mean distance to the cluster centroid, excluding noise; all-noise draws were omitted from the radius average. Silhouette was calculated after excluding noise and averaged only over draws containing at least two non-noise clusters and more assigned points than clusters; otherwise it was recorded as NA. The number of valid silhouette draws is shown for each projection. High-noise draws were retained when silhouette was defined, so scores describe the assigned subset and should be interpreted alongside noise fractions. Radius units depend on the projection and preprocessing.

## Outputs and provenance

Each organism produces a full-precision table with valid counts and descriptive SDs, a Word TSV, a Markdown table, all draw-level metrics, sampled IDs/indices, cluster labels, PCA coordinates and a validation manifest. Input hashes and row alignment are checked before computation; inputs are rehashed afterward. The aggregation step combines the three species tables.
