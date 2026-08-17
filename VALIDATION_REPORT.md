# Repository validation report

Validation performed on 2026-08-17 before packaging the GitHub release.

## Static checks

- all repository Python entry points compiled successfully with `py_compile`;
- all Bash, SLURM and SBATCH files passed `bash -n`;
- the publication-figure generator imports successfully and exposes its CLI.

## Unit tests

- leakage-controlled validation: **3/3 passed**;
- post-WEKA robustness: **6/6 passed**;
- network/candidate-selection module: **3/3 passed**.

## Canonical-configuration checks

- leakage branches: `legacy_global`, `nested_matched` only;
- leakage workload: 11 datasets × 10 seeds × 5 folds × 2 branches;
- K. pneumoniae +5/+6 historical pools: **60 / 60**;
- post-WEKA package version: **2.1.0**;
- final UMAP robustness fingerprints distinguish
  `geometry_evaluation=all_candidates` from
  `charge_dependent_evaluation=fixed_charge_balanced_subset`;
- the supplied publication-figure package is included with its exact source
  tables and rendered outputs.

Large scientific input/output datasets were intentionally not re-run during
this packaging validation; they remain part of the Zenodo/data-freeze step.
