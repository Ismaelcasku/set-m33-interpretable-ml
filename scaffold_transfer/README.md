# Linear-to-tetramer rank transfer

This utility reconstructs the numerical data for Table S8 and the rank-concordance columns of Table S7 from archived predictions. With the two archived network projections it also rebuilds the coverage columns. It does not fit or rerun the WEKA models.

```bash
python3 scaffold_transfer/summarize_predictions.py \
  SET_M33_S8_predictions.tar.gz tetramers_MIC_no_exact_training_overlap.csv new_results
```

The inputs are the archived E. coli out-of-fold predictions, the outputs for P. aeruginosa and K. pneumoniae from job 490536, and the corrected K. pneumoniae +5 output (batch5kleb60) from job 490662. The MIC input supplies the linear-arm sequences and observed tetramer MICs. Predictions are ranked within each species and arm-charge group, in ascending order. Exact ties receive the same rank. Strictly greater than twofold experimental MIC differences define eligible pairs. Predicted ties remain in the denominator and do not count as concordant.

For the complete S7, add `--coverage ecoli_arm_projection.csv paer_kpn_arm_projection.csv` to the command. Use job 490408 for E. coli and job 490534 for P. aeruginosa and K. pneumoniae. The E. coli file explicitly flags historical nodes, which are excluded. The other projection already contains only external arms. Coverage counts an external arm if its archived contact count is greater than zero.

The original cluster scripts are documented in [WEKA replay](weka_replay/README.md) and [coverage replay](../network_analysis/coverage_replay/README.md).
