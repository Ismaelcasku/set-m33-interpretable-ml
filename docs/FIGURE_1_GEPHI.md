# Figure 1: Gephi rendering boundary

Figure 1 is a representative physicochemical similarity network for the
E. coli +6 library at cosine-similarity threshold 0.70. The graph topology is
reconstructed by `network_analysis/scripts/run_network_analysis.py` and exported
as:

- `nodes_selected_threshold.csv`
- `edges_selected_threshold.csv`

The manuscript's final visualization was produced in **Gephi 0.10.1** using a
Fruchterman–Reingold layout. Because the final Gephi project/layout coordinates
were not part of the supplied code archive, the public repository does not claim
pixel-identical regeneration of Figure 1. The topology, threshold and plotting
variables are reproducible; the final layout is a visualization step.

## Recreate the figure

1. Generate `ecoli_6plus` network tables with `network_analysis/`.
2. Import `nodes_selected_threshold.csv` as the node table in Gephi using `ID`
   as the node identifier.
3. Import `edges_selected_threshold.csv` as an undirected edge table using
   `Source`, `Target`, and `Similarity` as the edge weight.
4. Apply the Fruchterman–Reingold layout to the graph and reuse the same layout
   for both panels.
5. Panel A: map node color categorically to `modularity_class`.
6. Panel B: map node color continuously to `predicted` MIC; lower and higher
   predictions correspond to the low/high ends of the manuscript color scale.
7. Keep node identity, edge set and coordinates unchanged between panels; only
   the node color encoding changes.

Louvain labels and PageRank exported by the Python reconstruction are provided
for reproducibility and candidate-selection provenance. Algorithm/software
versions can produce label-number permutations or small partition differences;
communities and layout geometry should not be interpreted as biological classes.
