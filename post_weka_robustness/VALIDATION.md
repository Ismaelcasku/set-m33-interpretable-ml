# Package validation

Validated before packaging on 2026-08-13.

## Scope verified

The package contains one dimensionality-reduction workflow only:

1. charge-specific signed SHAP values;
2. mean descriptor-by-charge SHAP dynamics;
3. zero filling of absent descriptor/charge combinations;
4. standardization of descriptor dynamics;
5. KMeans descriptor grouping (6/6/3 groups; seed 42; 10 initializations);
6. candidate-level mean SHAP within each descriptor group;
7. Euclidean UMAP without additional scaling (15 neighbors, minimum distance
   0.1, two components, seed 42, no subsampling).

No direct-SHAP/cosine UMAP implementation is present. Robustness analyses use
the same grouped SHAP representation and Euclidean distance.

## Static, dependency and unit validation

- all Python files compiled successfully;
- all Bash and SBATCH files passed `bash -n`;
- the six core unit tests passed, including corruption detection for reusable
  output markers, exact first-neighbor recovery and the distinction between
  conventional kNN overlap and Jaccard;
- the exact dependency set resolves together under Python 3.11;
- SHAP is pinned to 0.46.0 and `TreeExplainer` explicitly uses
  `feature_perturbation="tree_path_dependent"`;
- the workload resolves to 11 datasets, 220 Random Forest seed tasks,
  220 bootstrap tasks, 3 historical UMAP tasks and 27 UMAP robustness tasks
  (10 seeds per robustness task).

## Input validation

All eleven bundled CSVs passed their SHA-256, schema and numeric checks.

| Organism | Charge | Pool | Rows in file | Valid rows | Descriptors |
|---|---:|---:|---:|---:|
| E. coli | +4 | 10 | 250 | 250 | 94 |
| E. coli | +5 | 10 | 718 | 717 | 90 |
| E. coli | +6 | 10 | 1667 | 1667 | 90 |
| E. coli | +7 | 20 | 127 | 127 | 51 |
| P. aeruginosa | +4 | 10 | 3620 | 3620 | 93 |
| P. aeruginosa | +5 | 20 | 1199 | 1199 | 57 |
| P. aeruginosa | +6 | 10 | 526 | 526 | 98 |
| P. aeruginosa | +7 | 0 | 38 | 38 | 122 |
| K. pneumoniae | +4 | 10 | 798 | 798 | 84 |
| K. pneumoniae | +5 | 60 | 211 | 211 | 20 |
| K. pneumoniae | +6 | 60 | 165 | 165 | 19 |

The one incomplete *E. coli* +5 row is rejected explicitly, leaving 717 of
718 rows. It is not silently imputed.

## Historical reconstruction validation

Reference surrogate/SHAP and the complete historical dimensionality workflow
were executed successfully for all species using the bundled inputs.

| Organism | Candidates | Union descriptors | KMeans groups | Candidates by charge |
|---|---:|---:|---:|---|
| E. coli | 2,761 | 125 | 6 | 250 / 717 / 1,667 / 127 |
| P. aeruginosa | 5,383 | 200 | 6 | 3,620 / 1,199 / 526 / 38 |
| K. pneumoniae | 1,174 | 87 | 3 | 798 / 211 / 165 |

Each run generated the descriptor-by-charge dynamics, descriptor assignments,
grouped candidate matrix, Euclidean embedding and `UMAP_summary.csv`.

## End-to-end smoke validation

A reduced run completed successfully using the real bundled inputs:

- 2 repetitions of 2-fold surrogate CV for *E. coli* +4, including a reduced
  paired candidate bootstrap;
- full reference surrogate/SHAP for all four *E. coli* charge libraries;
- one alternative Random Forest seed and one fixed-seed bootstrap SHAP
  replicate;
- three Louvain seeds at the selected and adjacent HSPN thresholds;
- historical KMeans plus Euclidean UMAP for *E. coli*;
- robustness UMAP for two seeds on the same grouped SHAP matrix;
- partial final aggregation and manuscript-method generation;
- completed-output reuse checks for surrogate CV and historical UMAP.

Smoke outputs are deliberately excluded from the transfer archive and must not
be reported as scientific results.

## SLURM validation

A mocked submission produced the complete serial dependency chain:

`surrogate CV -> reference SHAP -> SHAP seeds -> SHAP bootstrap -> HSPN ->`
`historical UMAP -> UMAP robustness -> aggregation`.

Every task requests one CPU on `cpu20`; every array is throttled to at most 13
simultaneous tasks. Thus this project explicitly uses at most 13 CPUs and can
coexist with seven one-CPU WEKA jobs within the 20-CPU allocation. Matplotlib
and Numba caches are isolated per SLURM task.

## Statistical interpretation

- repeated-CV intervals are labelled as descriptive variability between reused
  partitions, not as independent inferential confidence intervals;
- candidate-level paired bootstrap intervals quantify surrogate fidelity;
- the SHAP bootstrap fixes the Random Forest seed at 42, isolating row-sampling
  variability from model-seed variability;
- SHAP ranking stability includes all pairwise Spearman correlations, top-10
  frequency and pairwise top-10 Jaccard overlap;
- UMAP/HDBSCAN results are stability diagnostics and are not interpreted as
  direct biological classes.

## Corrections in version 2.0.1

- `kneighbors(X=None)` is used with exactly `k` requested neighbors and no
  subsequent column removal, so metrics use neighbors 1 through 15 rather than
  neighbors 2 through 16;
- kNN preservation and between-run overlap use the explicitly documented
  conventional definition `|N1(i) ∩ N2(i)|/k`;
- the reference-SHAP continuation fingerprint includes
  `evaluation_max_rows`, preventing reuse after changing the stability subset
  size;
- the corrected kNN implementation identifier is part of every UMAP robustness
  fingerprint, invalidating any metrics produced by version 2.0.0 while leaving
  historical UMAP coordinates unchanged.


## Manuscript-aligned correction in version 2.1.0 — 2026-08-17

The final UMAP robustness implementation separates two questions that were
previously evaluated on the same charge-balanced subset:

- trustworthiness, exact 15-neighbor preservation, and between-run neighborhood
  overlap are now evaluated on the complete candidate library used to fit each
  UMAP embedding;
- charge-state silhouettes and HDBSCAN stability remain evaluated on the fixed
  charge-balanced subset.

The change does not alter the historical grouped-SHAP matrices or the primary
Euclidean UMAP coordinates. The UMAP robustness fingerprint includes
`geometry_evaluation=all_candidates`,
`charge_dependent_evaluation=fixed_charge_balanced_subset`, and the v3 kNN
implementation identifier, preventing reuse of the earlier robustness metrics.
The corrected 27-task UMAP grid and final aggregation were completed on the
cluster before the paper-aligned code freeze.
