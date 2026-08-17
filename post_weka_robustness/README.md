# SET-M33 historical post-WEKA pipeline and robustness audit

Cluster-first implementation beginning with the eleven archived candidate
tables containing WEKA predictions. It does not rerun CFS+BestFirst or WEKA and
does not use experimental MIC as a target.

The Python Random Forest is a surrogate trained to reproduce the archived WEKA
`predicted` column. TreeSHAP therefore explains the surrogate, not serialized
WEKA trees. Its fidelity is quantified before SHAP is interpreted.

## Single dimensionality pipeline

This version contains only the workflow recovered from the old executable code:

```text
full SHAP by charge
→ mean signed SHAP for each descriptor and charge
→ zero-fill absent descriptor/charge combinations
→ StandardScaler on descriptor × charge dynamics
→ KMeans descriptor groups (6 / 6 / 3; seed 42)
→ per-candidate mean SHAP within each group
→ unscaled Euclidean UMAP (15 neighbors, min_dist 0.1, seed 42)
```

No direct-SHAP/cosine UMAP branch is present. The manuscript description of a
direct cosine projection must be corrected because it does not match the
recovered code. See `HISTORICAL_UMAP_PROVENANCE.md` for the evidence boundary.

## Analyses

1. Surrogate fidelity: full-fit R2, MAE, RMSE and Spearman; ten repetitions of
   shuffled five-fold CV with the RF seed fixed at 42; descriptive variation
   across partitions; paired candidate bootstrap intervals using mean OOF
   predictions.
2. SHAP stability: 20 independent RF seeds; 200 nonparametric bootstrap training
   samples with RF seed fixed at 42; empirical intervals for `mean(|SHAP|)`;
   reference and pairwise Spearman ranking stability; top-10 frequency and
   Jaccard stability. All models explain the same deterministic,
   response-stratified subset of at most 1,000 candidates.
3. HSPN sensitivity: historical cosine-similarity threshold sweep; selected
   threshold and +/-0.05; 100 Louvain seeds; modularity, partition ARI, PageRank
   rank correlation, candidate-selection frequency and top-set Jaccard. This is
   a Python sensitivity audit and does not replace original Gephi labels.
4. Historical UMAP reproduction: exact grouped Euclidean transformation and
   complete, unsubsampled reference embedding for each organism.
5. Robustness of that same UMAP pipeline: Euclidean UMAP over three neighbor
   settings, three `min_dist` values and ten seeds. Trustworthiness, exact kNN
   preservation and between-run neighborhood overlap are evaluated on the
   complete candidate library. Charge-label silhouettes in grouped SHAP, PCA
   and UMAP, together with per-charge HDBSCAN stability and radii, are evaluated
   on a fixed charge-balanced subset. Every kNN metric uses the first 15
   non-self neighbors and reports the conventional mean overlap
   `|N1(i) ∩ N2(i)| / 15`, not Jaccard.

The cluster outputs are algorithmic summaries and are not treated as direct
evidence of biological classes. The all-candidate geometry evaluation and the
charge-balanced charge-dependent evaluation are intentionally separated in the
final manuscript-aligned implementation.

## Input data

The GitHub repository does not redistribute the eleven candidate-prediction
CSVs. Their filenames, pools and thresholds are fixed in `config/tasks.csv`;
*K. pneumoniae* +5 and +6 use pool 60. Obtain the study data from the Zenodo
archive, point `SET_M33_DATA_ROOT` to the extracted `Supplementary Data`
directory, and use `data/input_manifest.sha256` to verify the exact input bytes.

## Installation on Gorina

From `~/bhome/ML/weka_repeated_cv`:

```bash
tar -xzf SET_M33_post_WEKA_historical_robustness_HPC_v2.1.tar.gz
cd set_m33_post_weka_robustness
bash setup_environment.sh --conda
source activate_env.sh
python scripts/preflight.py
```

Use `bash setup_environment.sh --venv` if conda/mamba is unavailable. Exact
package versions are pinned in both environment files.

## Smoke test

```bash
source activate_env.sh
bash scripts/run_smoke.sh
```

This uses reduced repetitions and writes to `results_smoke/`. It validates code
paths but must not be reported as a scientific result.

## Full SLURM run: genuine 13-CPU cap

```bash
cd ~/bhome/ML/weka_repeated_cv/set_m33_post_weka_robustness
export SET_M33_THROTTLE=13
bash submit_all.sh
```

Every array task requests one CPU in partition `cpu20`. Complete arrays are
chained with `afterok`, so only one project array can run at a time and the
throttle is a genuine global cap of 13 CPU for this project. If seven one-CPU
WEKA jobs remain active, the combined maximum is 20 CPU.

To lower the allocation:

```bash
export SET_M33_THROTTLE=8
bash submit_all.sh
```

Values above 13 are rejected. Job IDs are written to
`results/submission/job_ids.tsv` and can be inspected with:

```bash
bash status.sh
```

The sequence is:

```text
surrogate CV
→ reference surrogate/SHAP
→ RF-seed SHAP
→ bootstrap SHAP
→ HSPN
→ historical grouped UMAP
→ robustness of grouped Euclidean UMAP
→ aggregation
```

Each completed unit has a configuration/input fingerprint and a completion
marker containing the size and SHA-256 of every required output. The marker is
set to `running` before files are replaced. A unit is reused only if its
fingerprint, file sizes and hashes all match; incomplete, corrupt or mismatched
outputs are recalculated. Matplotlib and Numba caches are isolated per SLURM
task.

## Outputs

```text
results/
├── surrogate_reference/
├── surrogate_cv/
├── shap_reference/
├── shap_seeds/
├── shap_bootstrap/
├── hspn/
├── historical_umap/
│   └── <species>/
│       ├── descriptor_charge_dynamics_*.csv
│       ├── descriptor_groups.csv
│       ├── grouped_shap.csv.gz
│       ├── historical_embedding.csv
│       └── UMAP_summary.csv
├── umap_robustness/
└── summary/
    ├── tables/
    ├── figures/
    ├── manuscript_methods.md
    ├── manuscript_results_template.md
    └── run_manifest.json
```

The archived numerical outputs are sufficient to produce the final SHAP
barplots, charge heatmaps and UMAP figures after the cluster analysis without
rerunning the expensive models.

## Statistical and reproducibility boundary

- `TreeExplainer` explicitly uses
  `feature_perturbation="tree_path_dependent"`.
- RF randomness and bootstrap sampling uncertainty are evaluated separately.
- Repeated-CV partition distributions are labelled descriptive because the
  same candidates are reused. Paired candidate bootstrap supplies the reported
  metric intervals.
- HDBSCAN is explicitly configured with `min_cluster_size=5` and
  `min_samples=5`. These are documented audit parameters because the deposited
  notebook referring to the old Supplementary Table S1 omitted its
  `cluster_metrics` implementation.
- The incomplete final row in *E. coli* +5 is rejected, retaining 717 of 718
  physical records.
