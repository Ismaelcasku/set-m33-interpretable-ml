# Physicochemical similarity networks and candidate selection

This module contains the physicochemical similarity-network construction and
candidate-prioritization logic used in the study. It is kept separate from the
leakage and SHAP/UMAP audit modules because it operates on the final analog
prediction tables rather than on the validation or attribution workflows.

## Canonical network definition

For each target/charge library, the descriptors in the final analog-prediction
CSV are checked against the corresponding selected training pool. Nonconstant
descriptors are standardized with `StandardScaler`, pairwise cosine similarity
is calculated, and edges are retained at the registered threshold in
`config/tasks.csv`.

Canonical selected thresholds are:

- E. coli +4/+5/+6/+7: 0.55 / 0.70 / 0.70 / 0.70
- P. aeruginosa +4/+5/+6/+7: 0.80 / 0.80 / 0.75 / 0.70
- K. pneumoniae +4/+5/+6: 0.65 / 0.65 / 0.60

All analogs remain nodes, including isolates. The standalone reconstruction uses
NetworkX Louvain (seed 42), PageRank, degree, weighted degree and connected
components. It exports Gephi-compatible node and edge tables.

## Candidate ranking

The candidate-prioritization rule used in the study is reproduced as follows:

1. rank communities by mean predicted MIC (lower is better);
2. retain the best half of communities, rounding up;
3. within each retained community, retain the top half of nodes by PageRank,
   rounding up, using lower predicted MIC as the tie-break;
4. rank the retained candidates by predicted MIC.

This is a prioritization rule in model-defined descriptor space, not evidence of
biological mechanism.

## Run

Install the lightweight dependencies in `requirements.txt`, extract the Zenodo
`Supplementary Data` directory, and run for all 11 libraries:

```bash
python scripts/run_network_analysis.py \
  --data-root /path/to/Supplementary\ Data \
  --task all
```

For a single library:

```bash
python scripts/run_network_analysis.py \
  --data-root /path/to/Supplementary\ Data \
  --task ecoli_6plus
```

Add `--full-topology-sweep` to calculate modularity and clustering throughout
the 0.05–0.95 threshold sweep. This is slower for dense graphs.

## Outputs

Each task writes:

- `threshold_metrics.csv`
- `threshold_sensitivity.svg`
- `nodes_selected_threshold.csv`
- `edges_selected_threshold.csv`
- `candidates_ranked.csv`
- `selected_network_metrics.json`
- `excluded_rows.csv`

`all_selected_network_metrics.csv` summarizes all executed libraries.

## Figure 1 boundary

The final network visualization in the manuscript was rendered in Gephi 0.10.1
with a Fruchterman–Reingold layout. This module reproduces the graph topology
and exports the node/edge tables, with final layout rendering in Gephi. See `../docs/FIGURE_1_GEPHI.md`.


## Input data

Candidate-selection inputs: package 02 at https://doi.org/10.5281/zenodo.22014734. Use a separate data root for network selection and interpretability.
