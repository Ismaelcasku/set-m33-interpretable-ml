#!/usr/bin/env python3
"""Run one branch x dataset x outer-seed x outer-fold audit task."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import (
    atomic_to_csv,
    atomic_write_json,
    dataset_path,
    load_config,
    project_root_from_file,
    read_weka_predictions,
    run_command_to_log,
    write_numeric_arff,
)
from leakage_common import (
    audit_fold_dir,
    audit_fold_is_complete,
    audit_root,
    audit_task_from_id,
    build_random_forest_command,
    cross_validated_selection_command,
    extended_regression_metrics,
    features_for_pool,
    global_selection_dir,
    load_audit_config,
    load_audit_manifest,
    parse_cv_attribute_frequency,
    prepare_positive_dataset,
    total_audit_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def require_pool_approval(root: Path, audit_config: dict, signature: str) -> None:
    path = audit_root(root, audit_config) / "manifests" / "GLOBAL_POOLS_APPROVED.json"
    if not path.is_file():
        raise RuntimeError(
            "Global pools have not been validated. Run scripts/validate_global_selection.py first."
        )
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("state") != "approved" or approval.get("run_signature") != signature:
        raise RuntimeError("Global-pool approval is stale or invalid")


def selection_frame(
    selected_features: list[str],
    frequency: pd.DataFrame,
    branch: str,
    dataset: str,
    seed: int,
    fold: int,
    method: str,
    pool_label: int | None,
) -> pd.DataFrame:
    lookup = frequency.set_index("feature") if not frequency.empty else pd.DataFrame()
    rows = []
    for rank, feature in enumerate(selected_features, start=1):
        row = {
            "branch": branch,
            "dataset": dataset,
            "seed": seed,
            "fold": fold,
            "selection_method": method,
            "pool_label": pool_label,
            "feature_rank": rank,
            "feature": feature,
            "selected_in_inner_fits": np.nan,
            "total_inner_selection_fits": np.nan,
            "inner_selection_frequency": np.nan,
        }
        if not frequency.empty and feature in lookup.index:
            values = lookup.loc[feature]
            row.update(
                {
                    "selected_in_inner_fits": int(values["selected_in_fits"]),
                    "total_inner_selection_fits": int(values["total_selection_fits"]),
                    "inner_selection_frequency": float(values["selection_frequency"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    task_id = args.task_id
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Provide --task-id or run inside a SLURM array")
        task_id = int(raw)

    root = args.project_root.resolve()
    project_config = load_config(root)
    audit_config = load_audit_config(root)
    manifest = load_audit_manifest(root, audit_config)
    signature = str(manifest["run_signature"])
    require_pool_approval(root, audit_config, signature)
    n_tasks = total_audit_tasks(project_config, audit_config)
    spec = audit_task_from_id(project_config, audit_config, int(task_id))
    base = spec.base
    output_dir = audit_fold_dir(root, audit_config, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    if not args.force and audit_fold_is_complete(root, audit_config, spec, signature):
        print(
            f"REUSED task {spec.task_id + 1}/{n_tasks}: {spec.branch}, "
            f"{base.dataset_name}, seed {base.seed}, fold {base.fold}",
            flush=True,
        )
        return

    started = time.time()
    base_status = {
        "task_id": spec.task_id,
        "base_task_id": spec.base_task_id,
        "branch": spec.branch,
        "dataset": base.dataset_name,
        "seed": base.seed,
        "fold": base.fold,
        "run_signature": signature,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "started_at": now(),
        "state": "running",
    }
    atomic_write_json(base_status, status_path)
    print(
        f"START task {spec.task_id + 1}/{n_tasks}: {spec.branch}, "
        f"{base.dataset_name}, seed {base.seed}, fold {base.fold}",
        flush=True,
    )

    scratch_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir())).resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"weka_audit_{spec.task_id:04d}_", dir=scratch_parent))
    try:
        dataset = project_config["datasets"][base.dataset_index]
        data = prepare_positive_dataset(
            base.dataset_name, dataset_path(root, dataset), project_config
        )
        X = data["X"].reset_index(drop=True)
        y_raw = data["y_raw"].reset_index(drop=True)
        y_log10 = data["y_log10"].reset_index(drop=True)
        splitter = KFold(
            n_splits=int(project_config["n_splits"]),
            shuffle=True,
            random_state=base.seed,
        )
        train_idx, test_idx = list(splitter.split(X))[base.fold - 1]

        frequency = pd.DataFrame()
        pool_label: int | None = None
        if spec.branch == "legacy_global":
            global_dir = global_selection_dir(root, audit_config, base.dataset_name)
            configured = pd.read_csv(global_dir / "configured_pool_features.csv")
            frequency = pd.read_csv(global_dir / "selection_frequency.csv")
            selected_features = configured["feature"].astype(str).tolist()
            pool_label = int(audit_config["historical_pool_labels"][base.dataset_name])
            selection_method = "global_weka_cv_pool"
        elif spec.branch == "nested_matched":
            selection_arff = work_dir / "outer_train_raw_mic_for_cv_selection.arff"
            write_numeric_arff(
                selection_arff,
                X.iloc[train_idx],
                y_raw.iloc[train_idx],
                relation=(
                    f"{base.dataset_name}_seed{base.seed}_fold{base.fold}_"
                    "outer_train_raw_mic_cv_selection"
                ),
            )
            selection_log = output_dir / "cross_validated_attribute_selection.log"
            inner_folds = int(audit_config["inner_selection_folds"])
            run_command_to_log(
                cross_validated_selection_command(
                    root,
                    project_config,
                    selection_arff,
                    n_folds=inner_folds,
                    seed=int(audit_config["inner_selection_seed"]),
                ),
                selection_log,
            )
            frequency = parse_cv_attribute_frequency(
                selection_log.read_text(encoding="utf-8", errors="replace"),
                list(X.columns),
                inner_folds,
            )
            pool_label = int(audit_config["historical_pool_labels"][base.dataset_name])
            selected_features = features_for_pool(frequency, pool_label, list(X.columns))
            selection_method = "outer_train_weka_cv_pool"
        else:
            raise ValueError(f"Unsupported branch: {spec.branch}")

        if len(selected_features) != len(set(selected_features)):
            raise RuntimeError("The selected feature list contains duplicates")
        missing_features = sorted(set(selected_features) - set(X.columns))
        if missing_features:
            raise RuntimeError(f"Selected descriptors are absent from the matrix: {missing_features[:20]}")
        selected = selection_frame(
            selected_features,
            frequency,
            spec.branch,
            base.dataset_name,
            base.seed,
            base.fold,
            selection_method,
            pool_label,
        )

        train_arff = work_dir / "rf_train_log10_mic.arff"
        test_arff = work_dir / "rf_test_log10_mic.arff"
        raw_predictions = work_dir / "weka_predictions.csv"
        write_numeric_arff(
            train_arff,
            X.iloc[train_idx][selected_features],
            y_log10.iloc[train_idx],
            relation=f"{base.dataset_name}_{spec.branch}_rf_train_log10_mic",
        )
        write_numeric_arff(
            test_arff,
            X.iloc[test_idx][selected_features],
            y_log10.iloc[test_idx],
            relation=f"{base.dataset_name}_{spec.branch}_rf_test_log10_mic",
        )
        classifier_log = output_dir / "random_forest.log"
        run_command_to_log(
            build_random_forest_command(
                root,
                project_config,
                train_arff,
                test_arff,
                raw_predictions,
                rf_seed=base.seed,
            ),
            classifier_log,
        )
        actual_weka, predicted = read_weka_predictions(
            raw_predictions, expected_n=len(test_idx)
        )
        expected_actual = y_log10.iloc[test_idx].to_numpy(dtype=float)
        if actual_weka is not None and np.isfinite(actual_weka).all():
            if not np.allclose(actual_weka, expected_actual, rtol=1e-8, atol=1e-10):
                raise ValueError("WEKA actual values/order do not match the held-out fold")

        predictions = pd.DataFrame(
            {
                "branch": spec.branch,
                "dataset": base.dataset_name,
                "seed": base.seed,
                "fold": base.fold,
                "row_index": test_idx.astype(int),
                "source_pandas_row": data["source_pandas_rows"].iloc[test_idx].to_numpy(dtype=int),
                "source_csv_line": data["source_csv_lines"].iloc[test_idx].to_numpy(dtype=int),
                "sample_id": data["ids"].iloc[test_idx].to_numpy(),
                "actual_model_scale": expected_actual,
                "predicted_model_scale": predicted,
                "actual_mic_original": data["target_original"].iloc[test_idx].to_numpy(dtype=float),
                "predicted_mic_original": np.power(10.0, predicted),
            }
        )
        predictions["residual_model_scale"] = (
            predictions["actual_model_scale"] - predictions["predicted_model_scale"]
        )
        metrics = extended_regression_metrics(
            predictions["actual_model_scale"], predictions["predicted_model_scale"]
        )
        atomic_to_csv(predictions, output_dir / "oof_predictions.csv")
        atomic_to_csv(selected, output_dir / "selected_features.csv")
        atomic_write_json(
            {
                "task_id": spec.task_id,
                "branch": spec.branch,
                "dataset": base.dataset_name,
                "seed": base.seed,
                "fold": base.fold,
                "run_signature": signature,
                "selection_method": selection_method,
                "pool_label": pool_label,
                "metrics": metrics,
            },
            output_dir / "fold_metrics.json",
        )

        completed = {
            **base_status,
            "state": "complete",
            "completed_at": now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "n_rows_before_positive_filter": int(data["n_rows_before_positive_filter"]),
            "n_rows_excluded_nonpositive": int(data["n_rows_excluded_nonpositive"]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_descriptors": int(X.shape[1]),
            "n_selected_features": int(len(selected_features)),
            "selection_method": selection_method,
            "pool_label": pool_label,
        }
        atomic_write_json(completed, status_path)
        atomic_write_json(completed, output_dir / "complete.json")
        print(
            f"COMPLETE task {spec.task_id + 1}/{n_tasks} in "
            f"{(time.time() - started) / 60:.1f} min; selected {len(selected_features)} descriptors",
            flush=True,
        )
    except Exception as exc:
        failed = {
            **base_status,
            "state": "failed",
            "failed_at": now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(failed, status_path)
        (output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAILED task {spec.task_id}: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
