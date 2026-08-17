#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from setm33robust.config import load_config, load_tasks, umap_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=["task_count", "species_count", "shap_seed_count", "bootstrap_chunk_count", "umap_task_count"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    values = {
        "task_count": len(load_tasks()),
        "species_count": load_tasks()["species"].nunique(),
        "shap_seed_count": len(cfg["shap"]["rf_seeds"]),
        "bootstrap_chunk_count": math.ceil(int(cfg["shap"]["bootstrap_replicates"]) / int(cfg["shap"]["bootstrap_chunk_size"])),
        "umap_task_count": len(umap_jobs(cfg)),
    }
    print(values[args.field])


if __name__ == "__main__":
    main()
