# Historical network coverage

The preserved launchers and scripts are from the Gorina E. coli and P. aeruginosa/K. pneumoniae projection runs used for Table S7. The archived arm projections (jobs 490408 and 490534) are the inputs to the summary script in `scaffold_transfer/`. For E. coli, that summary excludes historical nodes. The other projection already contains only external arms.

- `ecoli/`: `project_ecoli.slurm` runs `project_ecoli.py` with the original `inputs/` staging layout. The original revision export contains those inputs under `projection/inputs/`.
- `paer_kpn/`: the `run_no_pip.slurm` variant records package versions by importing them. It also reads `run.slurm` for a historical checksum; keep both files together. It expects the original `inputs/` layout and the SHAP reference root as its first argument. Only the arm projection is used for coverage in the paper.

These are preserved cluster scripts. Rebuilding Table S7 from archived outputs does not require rerunning either launcher.
