# Paper/data freeze checklist

- [x] Leakage audit contains 11 datasets and only `legacy_global` / `nested_matched`.
- [x] K. pneumoniae +5 and +6 historical pools are pool60 / pool60.
- [x] Network/candidate-selection logic is extracted into a standalone module.
- [x] Post-WEKA UMAP geometry metrics use all candidates; charge-dependent metrics use the balanced subset.
- [x] Figure 2 / Figure 3 generation package and exact plotting source data are included.
- [x] Figure 1 Gephi rendering boundary is documented.
- [ ] Choose an explicit software license if open reuse is intended.
- [ ] Create the public GitHub repository and add its URL to `CITATION.cff`.
- [ ] Upload final leakage-audit numerical outputs to the new Zenodo version.
- [ ] Upload final surrogate/SHAP/HSPN/historical-UMAP/UMAP-robustness outputs to Zenodo.
- [ ] Verify final Zenodo hashes and README against the frozen GitHub `v1.0.0` tag.
- [ ] Update manuscript Data Availability with both GitHub and Zenodo locations.
