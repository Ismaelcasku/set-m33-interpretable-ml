#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from setm33robust.config import cpu_count, load_config, resolve_data_root, resolve_results_root, task_at
from setm33robust.io import (
    analysis_fingerprint, begin_output, completed_output, dataset_path,
    load_prediction, sha256_file, write_completed_output,
)
from setm33robust.metrics import percentile_ci, regression_metrics
from setm33robust.models import make_surrogate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--candidate-bootstrap-replicates", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    task = task_at(args.task_index)
    data_root = resolve_data_root(cfg)
    results = resolve_results_root(cfg)
    frame, features, _ = load_prediction(data_root, task)
    X = frame[features].to_numpy(dtype=float)
    y = frame["predicted"].to_numpy(dtype=float)
    section = cfg["surrogate"]
    repeats = int(args.repeats or section["n_repeats"])
    splits = int(args.splits or section["n_splits"])
    candidate_bootstrap_reps = int(
        args.candidate_bootstrap_replicates or section["candidate_bootstrap_replicates"]
    )
    if splits > len(frame):
        raise ValueError("number of folds exceeds number of rows")
    base_seed = int(cfg["base_seed"])
    fingerprint = analysis_fingerprint({
        "stage": "surrogate_cv", "key": task["key"],
        "input_sha256": sha256_file(dataset_path(data_root, task)),
        "config": {
            "n_estimators": section["n_estimators"], "max_depth": section["max_depth"],
            "rf_seed": section["rf_seed"], "repeats": repeats, "splits": splits,
            "candidate_bootstrap_replicates": candidate_bootstrap_reps,
        },
    })
    out = results / "surrogate_cv" / str(task["key"])
    required = [
        "fold_metrics.csv", "repeat_metrics.csv", "oof_predictions.csv.gz",
        "partition_variability_summary.csv", "candidate_bootstrap_summary.csv",
        "candidate_bootstrap_metrics.csv.gz", "weka_vs_oof_surrogate.png",
    ]
    if completed_output(out, fingerprint, required):
        print(f"skip completed surrogate CV: {task['key']}")
        return
    out.mkdir(parents=True, exist_ok=True)
    begin_output(out, fingerprint, {"key": task["key"]})
    fold_rows, repeat_rows, prediction_rows = [], [], []
    for repeat in range(repeats):
        splitter = KFold(n_splits=splits, shuffle=True, random_state=base_seed + repeat)
        oof = np.full(len(y), np.nan, dtype=float)
        for fold, (train, test) in enumerate(splitter.split(X)):
            # Fixed model seed isolates variability due to the repeated partitions.
            model_seed = int(section["rf_seed"])
            model = make_surrogate(section, model_seed, cpu_count())
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            oof[test] = pred
            fold_rows.append({
                "key": task["key"], "repeat": repeat, "fold": fold,
                "n_train": len(train), "n_test": len(test), "model_seed": model_seed,
                **regression_metrics(y[test], pred),
            })
            prediction_rows.extend({
                "key": task["key"], "repeat": repeat, "fold": fold,
                "row_index": int(idx), "ID": frame.iloc[idx]["ID"],
                "weka_predicted": float(y[idx]), "surrogate_oof": float(value),
            } for idx, value in zip(test, pred))
        repeat_rows.append({
            "key": task["key"], "repeat": repeat, "n": len(y),
            **regression_metrics(y, oof),
        })
    folds = pd.DataFrame(fold_rows)
    repeats_df = pd.DataFrame(repeat_rows)
    predictions = pd.DataFrame(prediction_rows)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    repeats_df.to_csv(out / "repeat_metrics.csv", index=False)
    predictions.to_csv(out / "oof_predictions.csv.gz", index=False, compression="gzip")
    partition_summary = []
    for metric in ["r2", "mae", "rmse", "spearman"]:
        values = repeats_df[metric].to_numpy(dtype=float)
        _, low, high = percentile_ci(values)
        partition_summary.append({
            "metric": metric, "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "partition_interval_2p5": low, "partition_interval_97p5": high,
            "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "n_repeats": repeats,
            "interpretation": "descriptive variability across reused-data partitions",
        })
    pd.DataFrame(partition_summary).to_csv(out / "partition_variability_summary.csv", index=False)
    averaged = predictions.groupby("row_index", sort=True).agg(
        weka_predicted=("weka_predicted", "first"), surrogate_oof=("surrogate_oof", "mean")
    ).reset_index()
    rng = np.random.default_rng(base_seed + int(task["task_index"]) * 10_000)
    bootstrap_rows = []
    true = averaged["weka_predicted"].to_numpy(dtype=float)
    predicted = averaged["surrogate_oof"].to_numpy(dtype=float)
    for replicate in range(candidate_bootstrap_reps):
        sampled = rng.integers(0, len(averaged), size=len(averaged))
        bootstrap_rows.append({
            "replicate": replicate,
            **regression_metrics(true[sampled], predicted[sampled]),
        })
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(out / "candidate_bootstrap_metrics.csv.gz", index=False, compression="gzip")
    candidate_summary = []
    point = regression_metrics(true, predicted)
    for metric in ["r2", "mae", "rmse", "spearman"]:
        _, low, high = percentile_ci(bootstrap_df[metric])
        candidate_summary.append({
            "metric": metric, "point_estimate": point[metric],
            "ci95_low": low, "ci95_high": high,
            "bootstrap_replicates": candidate_bootstrap_reps,
            "resampling_unit": "candidate paired archived-WEKA/mean-OOF prediction",
        })
    pd.DataFrame(candidate_summary).to_csv(out / "candidate_bootstrap_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(averaged["weka_predicted"], averaged["surrogate_oof"], s=16, alpha=0.55)
    lo = min(averaged[["weka_predicted", "surrogate_oof"]].min())
    hi = max(averaged[["weka_predicted", "surrogate_oof"]].max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    ax.set(xlabel="Archived WEKA predicted", ylabel="Mean out-of-fold surrogate prediction", title=str(task["key"]))
    fig.tight_layout()
    fig.savefig(out / "weka_vs_oof_surrogate.png", dpi=300)
    plt.close(fig)
    write_completed_output(out, fingerprint, required, {
        "key": task["key"],
        "repeats": repeats, "splits": splits, "n": len(y), "n_features": len(features),
        "rf_seed_fixed_across_folds": int(section["rf_seed"]),
        "candidate_bootstrap_replicates": candidate_bootstrap_reps,
    })
    print(f"completed surrogate CV: {task['key']}")


if __name__ == "__main__":
    main()
