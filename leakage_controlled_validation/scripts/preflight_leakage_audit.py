#!/usr/bin/env python3
"""Validate the positive-MIC cohort and create an immutable audit manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (
    atomic_to_csv,
    atomic_write_json,
    dataset_path,
    file_sha256,
    find_column,
    load_config,
    load_manifest,
    project_root_from_file,
    target_masks,
)
from leakage_common import (
    audit_manifest_path,
    audit_root,
    audit_run_signature,
    audit_signature_payload,
    audit_task_from_id,
    load_audit_config,
    total_audit_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    project_config = load_config(root)
    audit_config = load_audit_config(root)
    original_manifest = load_manifest(root)
    signature = audit_run_signature(project_config, audit_config, original_manifest)
    output_root = audit_root(root, audit_config)
    manifests_dir = output_root / "manifests"

    dataset_names = [str(item["name"]) for item in project_config["datasets"]]
    pool_mapping = {
        str(key): int(value)
        for key, value in audit_config["historical_pool_labels"].items()
    }
    if set(pool_mapping) != set(dataset_names):
        missing = sorted(set(dataset_names) - set(pool_mapping))
        extra = sorted(set(pool_mapping) - set(dataset_names))
        raise ValueError(f"historical_pool_labels mismatch; missing={missing}, extra={extra}")
    allowed_labels = {int(value) for value in audit_config["pool_labels"]}
    invalid_labels = {name: label for name, label in pool_mapping.items() if label not in allowed_labels}
    if invalid_labels:
        raise ValueError(f"Unsupported historical pool labels: {invalid_labels}")

    previous_path = audit_manifest_path(root, audit_config)
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != signature:
            populated = any(
                path.name not in {"audit_manifest.json"}
                for path in output_root.rglob("*")
                if path.is_file()
            )
            if populated:
                raise RuntimeError(
                    f"{output_root} contains results for a different audit signature. "
                    "Keep it intact and choose a new audit_name before continuing."
                )

    original_inputs = {
        str(row["dataset"]): row for row in original_manifest.get("inputs", [])
    }
    cohort_rows = []
    excluded_rows = []
    for position, dataset in enumerate(project_config["datasets"], start=1):
        name = str(dataset["name"])
        path = dataset_path(root, dataset)
        print(f"[{position}/{len(dataset_names)}] positive-MIC audit: {name}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(f"Missing input for {name}: {path}")
        current_sha = file_sha256(path)
        expected = original_inputs.get(name)
        if expected is None or current_sha != str(expected.get("sha256")):
            raise RuntimeError(
                f"{name}: current input SHA-256 does not match the completed first experiment"
            )

        header = pd.read_csv(path, sep=project_config["csv_separator"], nrows=0)
        target_col = find_column(header, project_config["target_column"], required=True)
        id_col = find_column(header, project_config["id_column"], required=False)
        usecols = [target_col] + ([id_col] if id_col is not None else [])
        frame = pd.read_csv(
            path,
            sep=project_config["csv_separator"],
            usecols=usecols,
            low_memory=False,
        )
        parsed, missing, non_numeric = target_masks(frame[target_col])
        if non_numeric.any():
            bad_lines = (frame.index[non_numeric] + 2).tolist()[:20]
            raise ValueError(f"{name}: non-numeric MIC values at CSV lines {bad_lines}")
        included_before_positive = ~missing
        nonpositive = included_before_positive & (parsed <= 0)
        positive = included_before_positive & (parsed > 0)
        if int(positive.sum()) < int(project_config["n_splits"]):
            raise ValueError(f"{name}: fewer than five positive-MIC observations remain")

        sample_ids = (
            frame[id_col].astype("string")
            if id_col is not None
            else pd.Series([f"row_{index:05d}" for index in frame.index], dtype="string")
        )
        for index in frame.index[nonpositive]:
            excluded_rows.append(
                {
                    "dataset": name,
                    "pandas_row": int(index),
                    "csv_line": int(index) + 2,
                    "sample_id": sample_ids.iloc[index],
                    "mic_original": float(parsed.iloc[index]),
                    "reason": "MIC<=0",
                }
            )
        positive_values = parsed.loc[positive]
        cohort_rows.append(
            {
                "dataset": name,
                "file": str(dataset["file"]),
                "sha256": current_sha,
                "n_rows_source": int(len(frame)),
                "n_rows_missing_mic": int(missing.sum()),
                "n_rows_excluded_nonpositive_mic": int(nonpositive.sum()),
                "n_rows_positive_model_cohort": int(positive.sum()),
                "mic_min_positive": float(positive_values.min()),
                "mic_median_positive": float(positive_values.median()),
                "mic_max_positive": float(positive_values.max()),
                "historical_pool_label": pool_mapping[name],
            }
        )

    tasks = []
    for task_id in range(total_audit_tasks(project_config, audit_config)):
        spec = audit_task_from_id(project_config, audit_config, task_id)
        tasks.append(
            {
                "task_id": spec.task_id,
                "branch_index": spec.branch_index,
                "branch": spec.branch,
                "base_task_id": spec.base_task_id,
                "dataset_index": spec.base.dataset_index,
                "dataset": spec.base.dataset_name,
                "seed_index": spec.base.seed_index,
                "seed": spec.base.seed,
                "fold": spec.base.fold,
            }
        )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "run_signature": signature,
        "original_run_signature": str(original_manifest["run_signature"]),
        "method": {
            "cohort": "Exclude missing MIC and MIC <= 0 before any split, selection, or model fit",
            "selection_target": "raw MIC",
            "model_target": "log10(MIC)",
            "outer_evaluation": "10 seeds x 5 shuffled KFold splits",
            "legacy_global": "WEKA CFS+BestFirst 5-fold CV seed 1 on the complete positive cohort; fixed historical pool applied to every outer fold",
            "nested_matched": "The same WEKA CFS+BestFirst 5-fold CV seed 1 and pool construction repeated inside each outer training fold",
            "regressor": f"WEKA RandomForest, {int(project_config['n_trees'])} trees, seed matched to outer repetition",
            "pool_rule": audit_config["pool_rule"],
            "pool_rule_evidence": "inferred; global pool sizes must pass the documented historical-range check before a full run",
        },
        "signature_payload": audit_signature_payload(
            project_config, audit_config, original_manifest
        ),
        "project_configuration": project_config,
        "audit_configuration": audit_config,
        "cohorts": cohort_rows,
        "n_tasks": len(tasks),
        "original_csv_files_modified": False,
    }
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") == signature:
            manifest["created_at"] = previous.get("created_at", manifest["created_at"])

    atomic_to_csv(pd.DataFrame(cohort_rows), manifests_dir / "positive_mic_cohort_audit.csv")
    excluded_columns = [
        "dataset",
        "pandas_row",
        "csv_line",
        "sample_id",
        "mic_original",
        "reason",
    ]
    atomic_to_csv(
        pd.DataFrame(excluded_rows, columns=excluded_columns),
        manifests_dir / "excluded_nonpositive_mic.csv",
    )
    atomic_to_csv(pd.DataFrame(tasks), manifests_dir / "audit_task_manifest.csv")
    atomic_write_json(manifest, previous_path)
    print(
        f"PREFLIGHT COMPLETE: {len(cohort_rows)} datasets, {len(tasks)} fold tasks, "
        f"{len(excluded_rows)} MIC<=0 rows excluded.",
        flush=True,
    )
    print(f"Audit signature: {signature}", flush=True)


if __name__ == "__main__":
    main()
