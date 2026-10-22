# Reproducing the analyses

E. coli +5 contains 1,874 candidates. The joint E. coli UMAP contains 3,918 candidates. The workflow below fits the +5 surrogate, calculates its SHAP values and projects the joint E. coli grouped-SHAP matrix. The other charge-specific SHAP arrays and other-organism projections are loaded from package 05.

Use the pinned post-WEKA environment. Extract current packages 02 and 05 from the Zenodo data archive (concept DOI 10.5281/zenodo.20530642). The candidate root below is the nested `Candidate_libraries` folder in package 02. Run from `post_weka_robustness/`.

```bash
python scripts/prepare_complete_library.py \
  --candidate-root /path/to/Candidate_libraries \
  --archive-root /path/to/Post_WEKA_robustness \
  --work /path/to/analysis_work
export SET_M33_DATA_ROOT='/path/to/analysis_work/data/Supplementary Data'
export SET_M33_RESULTS_ROOT='/path/to/analysis_work/results'
python scripts/preflight.py
python scripts/run_surrogate_cv.py --task-index 1
python scripts/run_shap_reference.py --task-index 1
for seed in {0..19}; do
  python scripts/run_shap_seed.py --task-index 1 --seed-index "$seed"
done
for chunk in {0..19}; do
  python scripts/run_shap_bootstrap.py --task-index 1 --chunk-index "$chunk"
done
python scripts/run_historical_umap.py --species-index 0
for task in {0..8}; do
  python scripts/run_umap_robustness.py --umap-task-index "$task"
done
```

The loops run sequentially and can also be submitted as cluster jobs. HSPN and candidate selection use package 02 from DOI 10.5281/zenodo.22014734 in a separate data root. Their execution instructions are in the module READMEs. The `submit_all.sh` launcher executes all stages, including HSPN.

The aggregation script requires robustness outputs for all organisms. Combined summary tables are provided in package 05. S2 tables are calculated separately below.

## Tables S2.1–S2.3

Prepare the SHAP arrays and primary projection coordinates from deposited package 05:

```bash
python s2_tables/prepare_inputs.py \
  --archive-root /path/to/Post_WEKA_robustness \
  --work /path/to/new_s2_work
for species in ecoli paeruginosa kpneumoniae; do
  python s2_tables/recalculate_s2.py --bundle /path/to/new_s2_work \
    --species "$species" --out "/path/to/new_s2_work/results/$species"
done
python s2_tables/aggregate.py --work /path/to/new_s2_work
```

`s2_tables/METHODS.md` specifies the calculation: full descriptor-level SHAP participation ratio, joint unscaled two-component PCA, fixed primary UMAP coordinates and 20 charge-balanced draws. Noise points are excluded from silhouettes; invalid runs return NA and are omitted from silhouette means. Cluster-count and noise means include every draw. Majority-noise runs are retained when the silhouette is mathematically valid. Valid-run counts are reported.

## Figures

The current generator and rendered figures are in `../publication_figures/`. Pass the deposited package-05 root, or a selective workspace's `results` root, to `--results-root`. No projection is fitted during plotting.
