# Original Gorina WEKA scripts

These source files preserve the upstream runs behind the archived linear-arm predictions.

| Directory | Source run | Role |
|---|---|---|
| `ecoli/` | `tetramer_calibration_wKd8lb/charge_matched_v2` | Charge-matched E. coli evaluation; the manuscript refers to the archived corrected output, not a new fit. |
| `paer_kpn/` | `paer_kpn_revision_b2LnwG/rf_tetramers_HVFVXh`, job 490536 | Projection and predictions for P. aeruginosa and K. pneumoniae other than +5. |
| `kpn5_correction/` | `paer_kpn_revision_b2LnwG/kpn5_compare_Al3sKz`, job 490662 | Corrected K. pneumoniae +5 output from batch5kleb60. The corrected launcher is `compare_fixed.slurm`; it still hashes `compare.slurm`, which is kept as provenance. |

The launchers require the original staged inputs, the appropriate WEKA JAR and frozen models, and a compatible cluster environment. Model paths are inventoried in the S6–S8 data archive. These directories are not self-contained rerun packages. To regenerate the published tables from archived predictions, use the command in the parent directory.
