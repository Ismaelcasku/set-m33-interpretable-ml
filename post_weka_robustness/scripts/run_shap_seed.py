#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setm33robust.config import cpu_count, load_config, resolve_data_root, resolve_results_root, task_at
from setm33robust.io import (
    analysis_fingerprint, begin_files, completed_files, dataset_path,
    load_prediction, quantile_subset_indices, sha256_file,
    write_completed_files,
)
from setm33robust.metrics import regression_metrics
from setm33robust.models import make_surrogate, tree_shap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--evaluation-max-rows", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = list(cfg["shap"]["rf_seeds"])
    if args.seed_index < 0 or args.seed_index >= len(seeds):
        raise IndexError("seed-index outside configured rf_seeds")
    seed = int(seeds[args.seed_index])
    task = task_at(args.task_index)
    data_root = resolve_data_root(cfg)
    frame, features, _ = load_prediction(data_root, task)
    X = frame[features].to_numpy(dtype=float)
    y = frame["predicted"].to_numpy(dtype=float)
    max_rows = int(args.evaluation_max_rows or cfg["shap"]["evaluation_max_rows"])
    eval_idx = quantile_subset_indices(y, max_rows, int(cfg["base_seed"]))
    fingerprint = analysis_fingerprint({
        "stage": "shap_seed", "key": task["key"], "seed_index": args.seed_index,
        "rf_seed": seed, "evaluation_max_rows": max_rows,
        "input_sha256": sha256_file(dataset_path(data_root, task)),
        "surrogate": {
            "n_estimators": cfg["surrogate"]["n_estimators"],
            "max_depth": cfg["surrogate"]["max_depth"],
        },
        "tree_explainer_feature_perturbation": "tree_path_dependent",
    })
    out = resolve_results_root(cfg) / "shap_seeds" / str(task["key"])
    csv_path = out / f"seed_{args.seed_index:03d}.csv"
    marker_path = out / f"seed_{args.seed_index:03d}.json"
    if completed_files(marker_path, fingerprint, [csv_path]):
        print(f"skip completed SHAP seed: {task['key']} seed={seed}")
        return
    begin_files(marker_path, fingerprint, {"key": task["key"], "rf_seed": seed})
    model = make_surrogate(cfg["surrogate"], seed, cpu_count())
    model.fit(X, y)
    values, _ = tree_shap(model, X[eval_idx])
    importance = pd.DataFrame({
        "key": task["key"], "seed_index": args.seed_index, "rf_seed": seed,
        "descriptor": features, "mean_abs_shap": np.mean(np.abs(values), axis=0),
    })
    out.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out / f"seed_{args.seed_index:03d}.csv", index=False)
    write_completed_files(marker_path, fingerprint, [csv_path], {
        "key": task["key"], "seed_index": args.seed_index,
        "rf_seed": seed, "n_evaluation": len(eval_idx),
        "surrogate_metrics": regression_metrics(y, model.predict(X)),
    })
    print(f"completed SHAP seed: {task['key']} seed={seed}")


if __name__ == "__main__":
    main()
