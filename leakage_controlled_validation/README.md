# Leakage-controlled WEKA validation

This module reproduces the paired leakage audit reported in the manuscript. It
compares the historical global feature-selection workflow with a matched nested
workflow while keeping the historical pool level fixed for each target/charge
dataset.

## Design

The modeling cohort is defined before splitting by excluding missing MIC values
and MIC values <= 0. Feature selection uses raw MIC, whereas the Random Forest
regressor is fitted and evaluated on log10(MIC). Both branches use the same ten
outer seeds, five shuffled outer folds, 100-tree WEKA Random Forest, and the same
historical pool label.

| Branch | Feature selection | Purpose |
|---|---|---|
| `legacy_global` | WEKA CfsSubsetEval + BestFirst, fivefold CV (seed 1), on the complete positive-MIC cohort | Reproduces global descriptor selection before outer evaluation |
| `nested_matched` | The same selection procedure recalculated inside each outer-training fold | Paired leakage-controlled counterfactual |

The audit contains **11 datasets x 10 seeds x 5 folds x 2 branches = 1,100
outer Random Forest fits**.

Historical pool labels are fixed in `config/leakage_audit.json`:

- E. coli: 10 / 10 / 10 / 20 for +4 / +5 / +6 / +7;
- P. aeruginosa: 10 / 20 / 10 / 0 for +4 / +5 / +6 / +7;
- K. pneumoniae: 10 / 60 / 60 for +4 / +5 / +6.

Only `legacy_global` and `nested_matched` are part of this repository. No
`nested_current` branch is included.

## Requirements

- Python >= 3.9
- Java
- WEKA 3.8.6 (`weka.jar` is not redistributed here)
- Python packages in `requirements.txt`

Create the environment with:

```bash
bash scripts/setup_python.sh
```

Place WEKA 3.8.6 at `software/weka.jar`, or set `WEKA_JAR` to an absolute path.

## Input data

The 11 descriptor CSVs are intentionally not stored on GitHub. Place them in
`input/` using the exact filenames listed in `config/project.json`. The public
data archive and numerical outputs are deposited on Zenodo; the top-level
repository README records the dataset DOI.

The workflow hashes all input CSVs during preflight. Do not edit the files after
preflight.

## Reproduction

From this directory:

```bash
bash scripts/setup_python.sh
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/preflight_leakage_audit.py
```

The first preflight creates the immutable input/run manifest used by the audit;
it does not require running the older repeated-CV experiment.

Run the small validation submission first:

```bash
scripts/submit_leakage_audit.sh --smoke
.venv/bin/python scripts/check_leakage_progress.py
```

After inspecting the global-pool validation, run the complete paired audit:

```bash
scripts/submit_leakage_audit.sh --full
```

The submission launches two arrays of 550 tasks and aggregates only after both
arrays finish successfully. Missing tasks can be resumed with:

```bash
scripts/submit_leakage_audit.sh --retry
```

To aggregate an already complete run:

```bash
scripts/submit_leakage_audit.sh --aggregate-only
```

## Main outputs

Final tables are written under `results/leakage_audit_v1/summary/`, including:

- `paper_table_general_metrics.csv`
- `metrics_by_seed.csv`
- `paired_metric_differences_by_seed.csv`
- `paired_comparisons_bootstrap_95ci.csv`
- `branch_metrics_bootstrap_95ci.csv`
- `mean_oof_prediction_per_sample.csv`
- `feature_selection_frequency.csv`
- `feature_jaccard_summary.csv`

MAE, RMSE and R2 are evaluated on log10(MIC). Median relative error and the
within-2x/4x rates are evaluated on the original MIC scale. The sensitivity
analysis excluding the operational 10,000 uM records is produced during final
aggregation.

## Interpretation boundary

This audit quantifies optimism caused by reusing globally selected descriptor
identities. Because each target/charge dataset retains its historical pool
label, it does not test whether the original choice of pool label itself was
optimistic.
