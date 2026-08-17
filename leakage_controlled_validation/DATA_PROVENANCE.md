# Data provenance

This module expects the 11 full-descriptor training CSVs named in
`config/project.json`. They are not tracked by Git. Input SHA-256 values are
recorded by `scripts/preflight.py` in `results/manifests/run_manifest.json`, and
the leakage audit refuses to proceed if the bytes differ from that manifest.

The canonical target/charge set comprises four E. coli datasets, four P.
aeruginosa datasets, and three K. pneumoniae datasets. The canonical historical
pool mapping is stored in `config/leakage_audit.json`; K. pneumoniae +5 and +6
both use pool 60.
