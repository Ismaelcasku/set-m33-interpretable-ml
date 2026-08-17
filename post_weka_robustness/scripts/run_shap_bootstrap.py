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
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--total-replicates", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--evaluation-max-rows", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    task = task_at(args.task_index)
    shap_cfg = cfg["shap"]
    total = int(args.total_replicates or shap_cfg["bootstrap_replicates"])
    chunk_size = int(args.chunk_size or shap_cfg["bootstrap_chunk_size"])
    start = args.chunk_index * chunk_size
    stop = min(total, start + chunk_size)
    if start < 0 or start >= total:
        raise IndexError(f"chunk {args.chunk_index} starts outside 0..{total-1}")
    frame, features, _ = load_prediction(resolve_data_root(cfg), task)
    X = frame[features].to_numpy(dtype=float)
    y = frame["predicted"].to_numpy(dtype=float)
    max_rows = int(args.evaluation_max_rows or shap_cfg["evaluation_max_rows"])
    eval_idx = quantile_subset_indices(y, max_rows, int(cfg["base_seed"]))
    rf_seed = int(cfg["surrogate"]["rf_seed"])
    fingerprint = analysis_fingerprint({
        "stage": "shap_bootstrap", "key": task["key"], "chunk": args.chunk_index,
        "input_sha256": sha256_file(dataset_path(resolve_data_root(cfg), task)),
        "replicate_start": start, "replicate_stop": stop, "rf_seed": rf_seed,
        "evaluation_max_rows": max_rows,
        "surrogate": {
            "n_estimators": cfg["surrogate"]["n_estimators"],
            "max_depth": cfg["surrogate"]["max_depth"],
        },
        "tree_explainer_feature_perturbation": "tree_path_dependent",
    })
    out = resolve_results_root(cfg) / "shap_bootstrap" / str(task["key"])
    importance_path = out / f"chunk_{args.chunk_index:03d}_importance.csv.gz"
    metrics_path = out / f"chunk_{args.chunk_index:03d}_fit_metrics.csv"
    marker_path = out / f"chunk_{args.chunk_index:03d}.json"
    required_paths = [importance_path, metrics_path]
    if completed_files(marker_path, fingerprint, required_paths):
        print(f"skip completed SHAP bootstrap: {task['key']} chunk={args.chunk_index}")
        return
    begin_files(marker_path, fingerprint, {
        "key": task["key"], "chunk_index": args.chunk_index,
    })
    rows, fit_rows = [], []
    for replicate in range(start, stop):
        bootstrap_seed = int(cfg["base_seed"]) + int(task["task_index"]) * 100_000 + replicate
        rng = np.random.default_rng(bootstrap_seed)
        sampled = rng.integers(0, len(y), size=len(y))
        # Fix RF randomness so this branch isolates sampling uncertainty.
        model = make_surrogate(cfg["surrogate"], rf_seed, cpu_count())
        model.fit(X[sampled], y[sampled])
        values, _ = tree_shap(model, X[eval_idx])
        rows.extend({
            "key": task["key"], "replicate": replicate,
            "bootstrap_seed": bootstrap_seed, "rf_seed": rf_seed,
            "descriptor": feature, "mean_abs_shap": float(value),
        } for feature, value in zip(features, np.mean(np.abs(values), axis=0)))
        fit_rows.append({
            "key": task["key"], "replicate": replicate,
            "bootstrap_seed": bootstrap_seed, "rf_seed": rf_seed,
            "unique_training_rows": int(len(np.unique(sampled))),
            **regression_metrics(y, model.predict(X)),
        })
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"chunk_{args.chunk_index:03d}_importance.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fit_rows).to_csv(out / f"chunk_{args.chunk_index:03d}_fit_metrics.csv", index=False)
    write_completed_files(marker_path, fingerprint, required_paths, {
        "key": task["key"], "chunk_index": args.chunk_index,
        "replicate_start": start, "replicate_stop_exclusive": stop,
        "n_evaluation": len(eval_idx), "rf_seed_fixed": rf_seed,
    })
    print(f"completed SHAP bootstrap: {task['key']} replicates={start}:{stop}")


if __name__ == "__main__":
    main()
