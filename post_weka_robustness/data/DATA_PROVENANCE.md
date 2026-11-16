# Prediction inputs

The eleven candidate CSVs are provided in Zenodo package 02. Each contains selected physicochemical descriptors, ID, sequence and the WEKA predicted response. E. coli +5 contains 1,874 candidates. `input_manifest.sha256` specifies the input checksums.

The historical K. pneumoniae +5 candidate CSV used predictions from the +4
pool-60 WEKA model. Interpretability now uses predictions from
`batch5kleb60.model` for the same 211 candidate IDs. The +5 surrogate and
SHAP results and the joint K. pneumoniae projection were regenerated. The corrected CSV and its analysis outputs are available in packages 02 and 05 of the [published data version](https://doi.org/10.5281/zenodo.24281130).

Candidate-selection and HSPN inputs are provided in package 02 at DOI 10.5281/zenodo.22014734 and use a separate data root. See `../REPRODUCING_ANALYSES.md` for preparation and execution.
