#!/usr/bin/env python3
"""Build historical global feature pools for one dataset."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (
    atomic_to_csv,
    atomic_write_json,
    dataset_path,
    load_config,
    parse_selected_attributes,
    project_root_from_file,
    run_command_to_log,
    write_numeric_arff,
)
from leakage_common import (
    cross_validated_selection_command,
    features_for_pool,
    global_selection_dir,
    global_selection_is_complete,
    load_audit_config,
    load_audit_manifest,
    parse_cv_attribute_frequency,
    prepare_positive_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def main() -> None:
    args = parse_args()
    value = args.dataset_index
    if value is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Provide --dataset-index or run inside a SLURM array")
        value = int(raw)

    root = args.project_root.resolve()
    project_config = load_config(root)
    audit_config = load_audit_config(root)
    audit_manifest = load_audit_manifest(root, audit_config)
    signature = str(audit_manifest["run_signature"])
    if not 0 <= int(value) < len(project_config["datasets"]):
        raise ValueError(f"dataset-index must be in [0, {len(project_config['datasets']) - 1}]")
    dataset = project_config["datasets"][int(value)]
    dataset_name = str(dataset["name"])
    output_dir = global_selection_dir(root, audit_config, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    if not args.force and global_selection_is_complete(
        root, audit_config, dataset_name, signature
    ):
        print(f"REUSED global selection: {dataset_name}", flush=True)
        return

    started = time.time()
    base_status = {
        "dataset_index": int(value),
        "dataset": dataset_name,
        "run_signature": signature,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "started_at": now(),
        "state": "running",
    }
    atomic_write_json(base_status, status_path)
    scratch_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir())).resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"weka_global_{dataset_name}_", dir=scratch_parent))
    try:
        data = prepare_positive_dataset(
            dataset_name, dataset_path(root, dataset), project_config
        )
        X = data["X"].reset_index(drop=True)
        y_raw = data["y_raw"].reset_index(drop=True)
        train_arff = work_dir / "positive_cohort_raw_mic.arff"
        write_numeric_arff(
            train_arff,
            X,
            y_raw,
            relation=f"{dataset_name}_global_positive_raw_mic",
        )
        selection_log = output_dir / "cross_validated_attribute_selection.log"
        n_folds = int(audit_config["inner_selection_folds"])
        selection_seed = int(audit_config["inner_selection_seed"])
        print(
            f"RUNNING {dataset_name}: WEKA CFS+BestFirst {n_folds}-fold CV, seed {selection_seed}",
            flush=True,
        )
        run_command_to_log(
            cross_validated_selection_command(
                root,
                project_config,
                train_arff,
                n_folds=n_folds,
                seed=selection_seed,
            ),
            selection_log,
        )
        output = selection_log.read_text(encoding="utf-8", errors="replace")
        frequency = parse_cv_attribute_frequency(output, list(X.columns), n_folds)
        frequency.insert(0, "dataset", dataset_name)
        atomic_to_csv(frequency, output_dir / "selection_frequency.csv")

        direct_features = parse_selected_attributes(output, list(X.columns))
        atomic_to_csv(
            pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "feature_rank": range(1, len(direct_features) + 1),
                    "feature": direct_features,
                }
            ),
            output_dir / "full_cohort_direct_selected_features.csv",
        )

        summaries = []
        pool_frames = {}
        for label in [int(item) for item in audit_config["pool_labels"]]:
            features = features_for_pool(frequency, label, list(X.columns))
            pool_frame = pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "pool_label": label,
                    "feature_rank": range(1, len(features) + 1),
                    "feature": features,
                }
            )
            pool_frames[label] = pool_frame
            atomic_to_csv(pool_frame, output_dir / f"pool_{label:02d}_features.csv")
            summaries.append(
                {
                    "dataset": dataset_name,
                    "pool_label": label,
                    "n_features": len(features),
                    "minimum_selection_count": int(
                        frequency.loc[
                            frequency["feature"].isin(features), "selected_in_fits"
                        ].min()
                    ),
                }
            )
        pool_label = int(audit_config["historical_pool_labels"][dataset_name])
        configured = pool_frames[pool_label].copy()
        configured["pool_role"] = "historical_configured"
        atomic_to_csv(configured, output_dir / "configured_pool_features.csv")
        atomic_to_csv(pd.DataFrame(summaries), output_dir / "pool_summary.csv")

        completed = {
            **base_status,
            "state": "complete",
            "completed_at": now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "n_rows_source": int(data["n_rows_before_positive_filter"]),
            "n_rows_excluded_nonpositive": int(data["n_rows_excluded_nonpositive"]),
            "n_rows_positive": int(data["n_rows_model"]),
            "n_descriptors": int(X.shape[1]),
            "n_direct_selected": int(len(direct_features)),
            "configured_pool_label": pool_label,
            "n_configured_features": int(len(configured)),
        }
        atomic_write_json(completed, status_path)
        atomic_write_json(completed, output_dir / "complete.json")
        print(
            f"COMPLETE {dataset_name}: configured pool {pool_label}, {len(configured)} features",
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
        print(f"FAILED {dataset_name}: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
