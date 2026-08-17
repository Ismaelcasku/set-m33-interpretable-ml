#!/usr/bin/env python3
"""Shared utilities for the paired legacy-versus-nested leakage audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from common import (
    TaskSpec,
    canonical_sha256,
    java_prefix,
    prepare_dataset,
    regression_metrics,
    selection_command,
    task_from_id,
    total_tasks,
)


AUDIT_REQUIRED_PREDICTION_COLUMNS = {
    "branch",
    "dataset",
    "seed",
    "fold",
    "row_index",
    "source_pandas_row",
    "source_csv_line",
    "sample_id",
    "actual_model_scale",
    "predicted_model_scale",
    "actual_mic_original",
    "predicted_mic_original",
    "residual_model_scale",
}


@dataclass(frozen=True)
class AuditTaskSpec:
    task_id: int
    branch_index: int
    branch: str
    base_task_id: int
    base: TaskSpec


def audit_config_path(project_root: Path) -> Path:
    return Path(project_root).resolve() / "config" / "leakage_audit.json"


def load_audit_config(project_root: Path) -> Dict[str, Any]:
    path = audit_config_path(project_root)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported leakage-audit schema: {config.get('schema_version')!r}")
    expected_branches = {"legacy_global", "nested_matched"}
    branches = [str(value) for value in config.get("branches", [])]
    if set(branches) != expected_branches or len(branches) != len(expected_branches):
        raise ValueError(f"branches must contain exactly {sorted(expected_branches)}")
    if config.get("selection_target") != "raw_mic":
        raise ValueError("This audit requires selection_target='raw_mic'")
    if config.get("model_target") != "log10_mic":
        raise ValueError("This audit requires model_target='log10_mic'")
    if not bool(config.get("exclude_nonpositive_mic")):
        raise ValueError("exclude_nonpositive_mic must be true")
    if int(config.get("inner_selection_folds", 0)) < 2:
        raise ValueError("inner_selection_folds must be at least 2")
    labels = [int(value) for value in config.get("pool_labels", [])]
    if not labels or any(value < 0 or value >= 100 for value in labels):
        raise ValueError("pool_labels must contain values in [0, 100)")
    if config.get("pool_rule") != "selection_frequency_percent_strictly_greater_than_label":
        raise ValueError("Unsupported historical pool rule")
    return config


def audit_root(project_root: Path, audit_config: Dict[str, Any]) -> Path:
    return Path(project_root).resolve() / "results" / str(audit_config["audit_name"])


def audit_manifest_path(project_root: Path, audit_config: Dict[str, Any]) -> Path:
    return audit_root(project_root, audit_config) / "manifests" / "audit_manifest.json"


def load_audit_manifest(project_root: Path, audit_config: Dict[str, Any]) -> Dict[str, Any]:
    path = audit_manifest_path(project_root, audit_config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/preflight_leakage_audit.py before submitting jobs."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def total_audit_tasks(project_config: Dict[str, Any], audit_config: Dict[str, Any]) -> int:
    return total_tasks(project_config) * len(audit_config["branches"])


def audit_task_from_id(
    project_config: Dict[str, Any], audit_config: Dict[str, Any], task_id: int
) -> AuditTaskSpec:
    per_branch = total_tasks(project_config)
    total = total_audit_tasks(project_config, audit_config)
    if not 0 <= int(task_id) < total:
        raise ValueError(f"task_id must be between 0 and {total - 1}; received {task_id}")
    branch_index, base_task_id = divmod(int(task_id), per_branch)
    branch = str(audit_config["branches"][branch_index])
    return AuditTaskSpec(
        task_id=int(task_id),
        branch_index=branch_index,
        branch=branch,
        base_task_id=base_task_id,
        base=task_from_id(project_config, base_task_id),
    )


def global_selection_dir(
    project_root: Path, audit_config: Dict[str, Any], dataset_name: str
) -> Path:
    return audit_root(project_root, audit_config) / "global_selection" / dataset_name


def audit_fold_dir(
    project_root: Path, audit_config: Dict[str, Any], spec: AuditTaskSpec
) -> Path:
    base = spec.base
    return (
        audit_root(project_root, audit_config)
        / "branches"
        / spec.branch
        / "folds"
        / base.dataset_name
        / f"seed_{base.seed:02d}"
        / f"fold_{base.fold:02d}"
    )


def prepare_positive_dataset(
    dataset_name: str, path: Path, project_config: Dict[str, Any]
) -> Dict[str, Any]:
    if project_config.get("target_transform") != "none":
        raise ValueError(
            "config/project.json must retain target_transform='none'; the audit applies log10 only after selection"
        )
    data = prepare_dataset(dataset_name, path, project_config)
    target_original = data["target_original"].astype(float).reset_index(drop=True)
    keep = target_original > 0
    excluded_nonpositive = int((~keep).sum())

    filtered: Dict[str, Any] = dict(data)
    for key in (
        "X",
        "target_original",
        "ids",
        "source_pandas_rows",
        "source_csv_lines",
    ):
        value = data[key]
        filtered[key] = value.loc[keep].reset_index(drop=True)
    filtered["y_raw"] = filtered["target_original"].astype(float).rename("target_raw_mic")
    filtered["y_log10"] = np.log10(filtered["target_original"].astype(float)).rename(
        "target_log10_mic"
    )
    filtered["n_rows_before_positive_filter"] = int(len(target_original))
    filtered["n_rows_excluded_nonpositive"] = excluded_nonpositive
    filtered["n_rows_model"] = int(keep.sum())
    if filtered["n_rows_model"] < int(project_config["n_splits"]):
        raise ValueError(f"{dataset_name}: too few positive-MIC rows remain")
    return filtered


def build_random_forest_command(
    project_root: Path,
    project_config: Dict[str, Any],
    train_arff: Path,
    test_arff: Path,
    prediction_csv: Path,
    rf_seed: int,
) -> List[str]:
    prediction_spec = (
        "weka.classifiers.evaluation.output.prediction.CSV "
        f'-file "{prediction_csv}" -decimals 16 -suppress'
    )
    return java_prefix(project_root, project_config) + [
        project_config["weka"]["random_forest_class"],
        "-t",
        str(train_arff),
        "-T",
        str(test_arff),
        "-c",
        "last",
        "-classifications",
        prediction_spec,
        "-I",
        str(project_config["n_trees"]),
        "-S",
        str(int(rf_seed)),
    ]


def cross_validated_selection_command(
    project_root: Path,
    project_config: Dict[str, Any],
    train_arff: Path,
    n_folds: int,
    seed: int,
) -> List[str]:
    """Build WEKA's native cross-validated attribute-selection command."""
    command = selection_command(project_root, project_config, train_arff)
    search_position = command.index("-s")
    return command[:search_position] + [
        "-x",
        str(int(n_folds)),
        "-n",
        str(int(seed)),
    ] + command[search_position:]


def parse_cv_attribute_frequency(
    output: str, feature_names: Sequence[str], expected_folds: int
) -> pd.DataFrame:
    """Parse WEKA's ``number of folds (%) attribute`` stability table."""
    import re

    marker = re.search(
        r"Attribute\s+selection\s+\d+\s+fold\s+cross-validation",
        output,
        flags=re.IGNORECASE,
    )
    if marker is None:
        raise ValueError("WEKA cross-validation attribute-selection table was not found")
    section = output[marker.end() :]
    row_pattern = re.compile(
        r"^\s*(\d+)\s*\(\s*([0-9.]+)\s*%\s*\)\s+(\d+)\s+.*$"
    )
    rows: List[Dict[str, Any]] = []
    for line in section.splitlines():
        match = row_pattern.match(line)
        if match is None:
            continue
        count = int(match.group(1))
        attribute_index = int(match.group(3))
        if not 1 <= attribute_index <= len(feature_names):
            continue
        if not 0 <= count <= int(expected_folds):
            raise ValueError(f"Invalid WEKA selection count in line: {line!r}")
        rows.append(
            {
                "feature": str(feature_names[attribute_index - 1]),
                "selected_in_fits": count,
                "total_selection_fits": int(expected_folds),
                "selection_frequency": count / int(expected_folds),
                "selection_frequency_percent": 100.0 * count / int(expected_folds),
                "feature_order": attribute_index - 1,
            }
        )
    if not rows:
        raise ValueError("No descriptor frequencies could be parsed from WEKA output")
    frequency = pd.DataFrame(rows)
    duplicates = frequency.loc[frequency["feature"].duplicated(), "feature"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate descriptor frequencies in WEKA output: {duplicates[:10]}")
    return frequency.sort_values(
        ["selected_in_fits", "feature_order"], ascending=[False, True]
    ).reset_index(drop=True)


def selection_frequency_table(
    selections: Sequence[Tuple[int, Sequence[str]]], feature_order: Sequence[str]
) -> pd.DataFrame:
    if not selections:
        raise ValueError("At least one feature-selection result is required")
    counts: Dict[str, int] = {}
    for _, features in selections:
        for feature in set(map(str, features)):
            counts[feature] = counts.get(feature, 0) + 1
    order = {str(feature): position for position, feature in enumerate(feature_order)}
    unknown = sorted(set(counts) - set(order))
    if unknown:
        raise ValueError(f"Selected features are absent from the descriptor matrix: {unknown[:10]}")
    n_fits = len(selections)
    rows = [
        {
            "feature": feature,
            "selected_in_fits": count,
            "total_selection_fits": n_fits,
            "selection_frequency": count / n_fits,
            "selection_frequency_percent": 100.0 * count / n_fits,
            "feature_order": order[feature],
        }
        for feature, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["selected_in_fits", "feature_order"], ascending=[False, True]
    ).reset_index(drop=True)


def features_for_pool(
    frequency: pd.DataFrame, pool_label: int, feature_order: Sequence[str]
) -> List[str]:
    label = int(pool_label)
    retained = set(
        frequency.loc[
            frequency["selection_frequency_percent"] > label,
            "feature",
        ].astype(str)
    )
    selected = [str(feature) for feature in feature_order if str(feature) in retained]
    if not selected:
        raise ValueError(f"Pool {label} is empty")
    return selected


def extended_regression_metrics(
    actual_log10: Iterable[float], predicted_log10: Iterable[float]
) -> Dict[str, float]:
    metrics = regression_metrics(actual_log10, predicted_log10, "log10")
    actual = 10 ** np.asarray(actual_log10, dtype=float)
    predicted = 10 ** np.asarray(predicted_log10, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    relative = np.abs(predicted[valid] - actual[valid]) / actual[valid]
    metrics["median_relative_error_percent"] = float(100 * np.median(relative))
    return metrics


def global_selection_is_complete(
    project_root: Path,
    audit_config: Dict[str, Any],
    dataset_name: str,
    run_signature: str,
) -> bool:
    directory = global_selection_dir(project_root, audit_config, dataset_name)
    complete_path = directory / "complete.json"
    configured_path = directory / "configured_pool_features.csv"
    if not (complete_path.is_file() and configured_path.is_file()):
        return False
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        features = pd.read_csv(configured_path)
    except Exception:
        return False
    return bool(
        complete.get("run_signature") == run_signature
        and complete.get("dataset") == dataset_name
        and len(features) > 0
        and "feature" in features.columns
    )


def audit_fold_is_complete(
    project_root: Path,
    audit_config: Dict[str, Any],
    spec: AuditTaskSpec,
    run_signature: str,
) -> bool:
    directory = audit_fold_dir(project_root, audit_config, spec)
    complete_path = directory / "complete.json"
    predictions_path = directory / "oof_predictions.csv"
    selected_path = directory / "selected_features.csv"
    if not (complete_path.is_file() and predictions_path.is_file() and selected_path.is_file()):
        return False
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path)
        selected = pd.read_csv(selected_path)
    except Exception:
        return False
    metadata_ok = (
        complete.get("run_signature") == run_signature
        and int(complete.get("task_id", -1)) == spec.task_id
        and complete.get("branch") == spec.branch
        and complete.get("dataset") == spec.base.dataset_name
        and int(complete.get("seed", -1)) == spec.base.seed
        and int(complete.get("fold", -1)) == spec.base.fold
    )
    predictions_ok = (
        AUDIT_REQUIRED_PREDICTION_COLUMNS.issubset(predictions.columns)
        and len(predictions) > 0
    )
    selected_ok = {"branch", "dataset", "seed", "fold", "feature"}.issubset(
        selected.columns
    ) and len(selected) > 0
    return bool(metadata_ok and predictions_ok and selected_ok)


def audit_signature_payload(
    project_config: Dict[str, Any],
    audit_config: Dict[str, Any],
    original_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "project_config": project_config,
        "audit_config": audit_config,
        "original_run_signature": original_manifest["run_signature"],
        "inputs": {
            str(row["dataset"]): str(row["sha256"])
            for row in original_manifest["inputs"]
        },
    }


def audit_run_signature(
    project_config: Dict[str, Any],
    audit_config: Dict[str, Any],
    original_manifest: Dict[str, Any],
) -> str:
    return canonical_sha256(
        audit_signature_payload(project_config, audit_config, original_manifest)
    )
