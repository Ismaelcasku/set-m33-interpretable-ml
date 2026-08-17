#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setm33robust.config import cpu_count, load_config, resolve_data_root, resolve_results_root, task_at
from setm33robust.io import (
    analysis_fingerprint, begin_files, completed_files, dataset_path,
    load_prediction, quantile_subset_indices, save_shap_reference, sha256_file,
    write_completed_files, write_json,
)
from setm33robust.metrics import regression_metrics
from setm33robust.models import make_surrogate, tree_shap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--evaluation-max-rows", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    task = task_at(args.task_index)
    data_root = resolve_data_root(cfg)
    results = resolve_results_root(cfg)
    frame, features, _ = load_prediction(data_root, task)
    X = frame[features].to_numpy(dtype=float)
    y = frame["predicted"].to_numpy(dtype=float)
    section = cfg["surrogate"]
    max_rows = int(args.evaluation_max_rows or cfg["shap"]["evaluation_max_rows"])
    fingerprint = analysis_fingerprint({
        "stage": "reference_shap", "key": task["key"],
        "input_sha256": sha256_file(dataset_path(data_root, task)),
        "config": {
            "n_estimators": section["n_estimators"], "max_depth": section["max_depth"],
            "rf_seed": section["rf_seed"],
            "evaluation_max_rows": max_rows,
            "tree_explainer_feature_perturbation": "tree_path_dependent",
        },
    })
    shap_out = results / "shap_reference" / str(task["key"])
    ref_out = results / "surrogate_reference" / str(task["key"])
    marker_path = shap_out / "run.json"
    required_paths = [
        shap_out / "shap_values.npy", shap_out / "features.json",
        shap_out / "rows.csv", shap_out / "expected_value.json",
        shap_out / "reference_importance.csv",
        shap_out / "stability_evaluation_rows.csv",
        ref_out / "surrogate.joblib", ref_out / "fitted_predictions.csv",
        ref_out / "metrics.json", ref_out / "weka_vs_fitted_surrogate.png",
    ]
    if completed_files(marker_path, fingerprint, required_paths):
        print(f"skip completed reference surrogate + SHAP: {task['key']}")
        return
    begin_files(marker_path, fingerprint, {"key": task["key"]})
    model = make_surrogate(section, int(section["rf_seed"]), cpu_count())
    model.fit(X, y)
    pred = model.predict(X)
    metrics = regression_metrics(y, pred)
    ref_out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ref_out / "surrogate.joblib", compress=3)
    pd.DataFrame({
        "ID": frame["ID"], "weka_predicted": y, "surrogate_fitted": pred,
        "residual": y - pred,
    }).to_csv(ref_out / "fitted_predictions.csv", index=False)
    write_json(ref_out / "metrics.json", metrics)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(y, pred, s=16, alpha=0.55)
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    ax.set(xlabel="Archived WEKA predicted", ylabel="Fitted surrogate prediction", title=str(task["key"]))
    ax.text(0.03, 0.97, f"R2={metrics['r2']:.3f}\nMAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}\nSpearman={metrics['spearman']:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(ref_out / "weka_vs_fitted_surrogate.png", dpi=300)
    plt.close(fig)

    shap_values, expected = tree_shap(model, X)
    rows = frame[["ID", "predicted", "sequence"]].copy()
    rows["charge"] = int(task["charge"])
    rows["species"] = str(task["species"])
    save_shap_reference(shap_out, shap_values, features, rows, expected)
    eval_idx = quantile_subset_indices(y, max_rows, int(cfg["base_seed"]))
    importance = pd.DataFrame({
        "descriptor": features,
        "mean_abs_shap_full": np.mean(np.abs(shap_values), axis=0),
        "mean_signed_shap_full": np.mean(shap_values, axis=0),
        "mean_abs_shap_stability_subset": np.mean(np.abs(shap_values[eval_idx]), axis=0),
        "mean_signed_shap_stability_subset": np.mean(shap_values[eval_idx], axis=0),
    }).sort_values("mean_abs_shap_stability_subset", ascending=False)
    importance.to_csv(shap_out / "reference_importance.csv", index=False)
    pd.DataFrame({"row_index": eval_idx, "ID": frame.iloc[eval_idx]["ID"].to_numpy()}).to_csv(shap_out / "stability_evaluation_rows.csv", index=False)
    write_completed_files(marker_path, fingerprint, required_paths, {
        "key": task["key"], "n": len(frame),
        "n_features": len(features), "n_stability_evaluation": len(eval_idx),
        "evaluation_max_rows": max_rows,
        "rf_seed": int(section["rf_seed"]), "surrogate_metrics": metrics,
    })
    print(f"completed reference surrogate + SHAP: {task['key']}")


if __name__ == "__main__":
    main()
