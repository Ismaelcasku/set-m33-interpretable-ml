# SET-M33 interpretable machine-learning analyses

Code supporting the manuscript **“Interpretable machine learning identifies
target-dependent predictive patterns for a dendrimeric antimicrobial peptide.”**

This repository is the paper-aligned **code release**. Large descriptor matrices,
WEKA artifacts and numerical analysis outputs are kept in the study archive on
Zenodo (current record: **10.5281/zenodo.20610591**) and will be updated with the
final revision outputs. GitHub contains executable code, configuration, tests,
figure-generation material and provenance documentation.

## Repository structure

```text
set-m33-interpretable-ml/
├── network_analysis/
│   └── physicochemical similarity networks and candidate prioritization
├── leakage_controlled_validation/
│   └── paired legacy_global vs nested_matched WEKA leakage audit
├── post_weka_robustness/
│   └── surrogate fidelity, SHAP robustness, HSPN sensitivity and historical UMAP
├── publication_figures/
│   └── exact Figure 2/Figure 3 plotting package, source data and rendered outputs
└── docs/
    └── data boundary, manuscript-code map, Figure 1/Gephi note and release guide
```

## Canonical analyses for the revised manuscript

### 1. Physicochemical similarity networks

`network_analysis/` reconstructs the 11 charge-defined similarity networks from
StandardScaler-normalized selected descriptors and cosine similarity, exports
Gephi-compatible node/edge tables, calculates graph annotations, and reproduces
the candidate-prioritization rule used in the study. The final Figure 1 layout remains a
Gephi visualization step; see `docs/FIGURE_1_GEPHI.md`.

### 2. Leakage-controlled validation

`leakage_controlled_validation/` contains the paired 11-dataset audit reported
in the revised manuscript. The public analysis has only two branches:

- `legacy_global`
- `nested_matched`

Both use identical 10-seed × 5-fold outer partitions. Historical pool levels are
fixed per target/charge dataset, including **pool60 for K. pneumoniae +5 and +6**.
The primary comparison therefore contains **1,100 outer Random Forest fits**.

### 3. Surrogate SHAP and historical UMAP robustness

`post_weka_robustness/` starts from the archived WEKA candidate predictions. The
Python Random Forest is a surrogate of the WEKA `predicted` column; SHAP explains
that surrogate rather than the serialized WEKA trees or a causal biological
mechanism.

The historical dimensionality workflow is:

```text
charge-specific signed SHAP
→ descriptor × charge mean signed SHAP
→ zero-fill absent descriptor/charge combinations
→ StandardScaler
→ KMeans descriptor groups (6 / 6 / 3)
→ candidate-level mean signed SHAP within each group
→ unscaled Euclidean UMAP
```

For the final robustness analysis, trustworthiness, exact 15-neighbor
preservation and between-run neighborhood overlap are evaluated on the **complete
candidate library**. Charge-state silhouettes and HDBSCAN stability are evaluated
separately on a **fixed charge-balanced subset**.

### 4. Publication figures

`publication_figures/` preserves the script used to generate Figure 2 (SHAP
heatmaps), Figure 3 (historical grouped Euclidean UMAP) and the supplementary
SHAP top-20 panels. It also contains the exact plotting source tables and the
rendered PNG/PDF/SVG outputs. The script does not refit models, recalculate SHAP
or rerun UMAP.

## Data and results

The study data archive is currently available at Zenodo DOI
**10.5281/zenodo.20610591**. See `docs/DATA_AND_ZENODO.md` for the intended final
GitHub/Zenodo boundary and expected data layouts.

## Reproducibility and interpretation boundary

Each executable module has its own README, configuration and tests. The analyses
are deliberately separated because they have different dependencies and because
some stages are historical reconstructions rather than one monolithic current
pipeline.

SHAP values are predictive attributions of surrogate models. Similarity-network
communities, UMAP neighborhoods and HDBSCAN clusters are algorithmic
representations. None of these outputs should be presented as direct evidence of
causal peptide–membrane mechanisms or biological classes.

## Citation

Use `CITATION.cff` for the software metadata. Please also cite the associated
manuscript and the Zenodo study-data record.

## License

No explicit open-source software license is assigned in this release. The code
is therefore distributed under the default copyright terms unless the authors
add a license before public release.
