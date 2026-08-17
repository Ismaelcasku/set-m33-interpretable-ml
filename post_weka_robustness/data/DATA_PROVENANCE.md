# Bundled input data

This directory contains only the eleven final analog-candidate CSV files used
as inputs to the post-WEKA analysis. They are byte-for-byte copies of the files
under `Supplementary Data/Test/Analogs predictions/` in the audited Zenodo
deposit.

Each table contains physicochemical descriptors, candidate identifier,
sequence, and the archived WEKA `predicted` response. No experimental MIC is
used by this package. `input_manifest.sha256` identifies the exact bytes.

The *E. coli* +5 CSV contains 718 physical rows but one incomplete final record;
the analysis retains the 717 valid records and exports the rejected row during
preflight.
