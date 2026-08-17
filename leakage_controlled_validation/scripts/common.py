#!/usr/bin/env python3
"""Shared, deterministic utilities for the leakage-controlled WEKA workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REQUIRED_PREDICTION_COLUMNS = {
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
class TaskSpec:
    task_id: int
    dataset_index: int
    dataset_name: str
    dataset_file: str
    seed_index: int
    seed: int
    fold: int


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(project_root: Path, config_path: Optional[Path] = None) -> Dict[str, Any]:
    project_root = Path(project_root).resolve()
    path = Path(config_path).resolve() if config_path else project_root / "config" / "project.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported configuration schema: {config.get('schema_version')!r}")
    if config.get("target_transform") not in {"none", "log10"}:
        raise ValueError("target_transform must be 'none' or 'log10'")
    if config.get("missing_target_policy") not in {"exclude", "error"}:
        raise ValueError("missing_target_policy must be 'exclude' or 'error'")
    if not config.get("datasets") or not config.get("seeds"):
        raise ValueError("datasets and seeds cannot be empty")
    if int(config.get("n_splits", 0)) < 2:
        raise ValueError("n_splits must be at least 2")
    return config


def total_tasks(config: Dict[str, Any]) -> int:
    return len(config["datasets"]) * len(config["seeds"]) * int(config["n_splits"])


def task_from_id(config: Dict[str, Any], task_id: int) -> TaskSpec:
    n_splits = int(config["n_splits"])
    seeds = [int(seed) for seed in config["seeds"]]
    per_dataset = len(seeds) * n_splits
    n_tasks = total_tasks(config)
    if not 0 <= int(task_id) < n_tasks:
        raise ValueError(f"task_id must be between 0 and {n_tasks - 1}; received {task_id}")
    dataset_index, within_dataset = divmod(int(task_id), per_dataset)
    seed_index, fold_zero = divmod(within_dataset, n_splits)
    dataset = config["datasets"][dataset_index]
    return TaskSpec(
        task_id=int(task_id),
        dataset_index=dataset_index,
        dataset_name=str(dataset["name"]),
        dataset_file=str(dataset["file"]),
        seed_index=seed_index,
        seed=seeds[seed_index],
        fold=fold_zero + 1,
    )


def dataset_path(project_root: Path, task_or_dataset: Any) -> Path:
    file_name = task_or_dataset.dataset_file if isinstance(task_or_dataset, TaskSpec) else task_or_dataset["file"]
    return Path(project_root).resolve() / "input" / str(file_name)


def fold_dir(project_root: Path, spec: TaskSpec) -> Path:
    return (
        Path(project_root).resolve()
        / "results"
        / "folds"
        / spec.dataset_name
        / f"seed_{spec.seed:02d}"
        / f"fold_{spec.fold:02d}"
    )


def final_model_dir(project_root: Path, dataset_name: str) -> Path:
    return Path(project_root).resolve() / "results" / "final_models" / dataset_name


def manifest_path(project_root: Path) -> Path:
    return Path(project_root).resolve() / "results" / "manifests" / "run_manifest.json"


def load_manifest(project_root: Path) -> Dict[str, Any]:
    path = manifest_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/preflight.py successfully before submitting jobs."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def find_column(frame: pd.DataFrame, requested: str, required: bool = True) -> Optional[str]:
    exact = [column for column in frame.columns if str(column) == requested]
    if exact:
        return exact[0]
    folded = [
        column
        for column in frame.columns
        if str(column).strip().casefold() == str(requested).casefold()
    ]
    if len(folded) == 1:
        return folded[0]
    if required:
        available = "\n".join(map(str, frame.columns))
        raise KeyError(f"Column {requested!r} was not found. Available columns:\n{available}")
    return None


def target_masks(raw_target: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    parsed = pd.to_numeric(raw_target, errors="coerce")
    raw_text = raw_target.astype("string").str.strip()
    blank = (raw_target.isna() | raw_text.isna() | raw_text.eq("")).fillna(True)
    missing = parsed.isna() & blank
    non_numeric = parsed.isna() & ~blank
    return parsed.astype(float), missing, non_numeric


def find_invalid_target_rows(
    dataset_name: str, path: Path, config: Dict[str, Any]
) -> pd.DataFrame:
    path = Path(path)
    header = pd.read_csv(path, sep=config["csv_separator"], nrows=0)
    target_col = find_column(header, config["target_column"], required=True)
    id_col = find_column(header, config["id_column"], required=False)
    usecols = [target_col] + ([id_col] if id_col is not None else [])
    frame = pd.read_csv(path, sep=config["csv_separator"], usecols=usecols, low_memory=False)
    _, missing, non_numeric = target_masks(frame[target_col])
    bad = missing | non_numeric
    columns = ["dataset", "pandas_row", "csv_line", "sample_id", "mic_raw", "problem"]
    if not bad.any():
        return pd.DataFrame(columns=columns)
    bad_index = frame.index[bad]
    if id_col is None:
        sample_ids = pd.Series([f"row_{i:05d}" for i in bad_index], index=bad_index)
    else:
        sample_ids = frame.loc[bad_index, id_col].astype("string")
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "pandas_row": bad_index.astype(int),
            "csv_line": bad_index.astype(int) + 2,
            "sample_id": sample_ids.astype("string").to_numpy(),
            "mic_raw": frame.loc[bad_index, target_col].astype("string").fillna("<NA>").to_numpy(),
            "problem": np.where(missing.loc[bad_index].to_numpy(), "missing", "non_numeric"),
        }
    )


def prepare_dataset(dataset_name: str, path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(path)
    frame = pd.read_csv(path, sep=config["csv_separator"], low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    n_rows_source = len(frame)
    target_col = find_column(frame, config["target_column"], required=True)
    id_col = find_column(frame, config["id_column"], required=False)
    _, missing_target, non_numeric_target = target_masks(frame[target_col])

    if non_numeric_target.any():
        details = find_invalid_target_rows(dataset_name, path, config)
        details = details.query("problem == 'non_numeric'").head(20).to_string(index=False)
        raise ValueError(f"{dataset_name}: non-numeric MIC values require manual review:\n{details}")
    if missing_target.any() and config["missing_target_policy"] == "error":
        details = find_invalid_target_rows(dataset_name, path, config)
        details = details.query("problem == 'missing'").head(20).to_string(index=False)
        raise ValueError(f"{dataset_name}: missing MIC values were found:\n{details}")

    if config["missing_target_policy"] == "exclude":
        frame = frame.loc[~missing_target].copy()
    source_pandas_rows = pd.Series(
        frame.index.to_numpy(dtype=int), name="source_pandas_row"
    ).reset_index(drop=True)
    source_csv_lines = (source_pandas_rows + 2).rename("source_csv_line")
    frame = frame.reset_index(drop=True)

    if id_col is None:
        ids = source_pandas_rows.map(lambda index: f"row_{index:05d}").rename("sample_id")
        id_source = "generated_from_source_row_number"
    else:
        ids = frame[id_col].astype(str).rename("sample_id")
        id_source = str(id_col)

    excluded = [target_col] + ([id_col] if id_col is not None else [])
    raw_descriptors = frame.drop(columns=excluded)
    if raw_descriptors.shape[1] == 0:
        raise ValueError(f"{dataset_name}: no descriptors remain after excluding ID and MIC")

    converted_columns: Dict[str, pd.Series] = {}
    bad_columns: List[Tuple[str, List[str]]] = []
    for column in raw_descriptors.columns:
        original = raw_descriptors[column]
        converted = pd.to_numeric(original, errors="coerce").astype(float)
        nonempty = original.notna() & original.astype("string").str.strip().ne("")
        introduced_missing = converted.isna() & nonempty
        if introduced_missing.any():
            bad_columns.append((str(column), original.loc[introduced_missing].astype(str).head(3).tolist()))
        converted_columns[str(column)] = converted
    descriptors = pd.DataFrame(converted_columns, index=frame.index).copy()
    if bad_columns:
        details = "\n".join(f"- {column}: {values}" for column, values in bad_columns[:20])
        raise TypeError(f"{dataset_name}: non-numeric descriptor values were found:\n{details}")

    target_original = pd.to_numeric(frame[target_col], errors="coerce").astype(float)
    if target_original.isna().any():
        raise RuntimeError(f"{dataset_name}: missing MIC remained after applying the policy")
    if np.isinf(descriptors.to_numpy(dtype=float)).any() or np.isinf(target_original.to_numpy()).any():
        raise ValueError(f"{dataset_name}: infinite values are not supported")

    if config["target_transform"] == "log10":
        if (target_original <= 0).any():
            raise ValueError(f"{dataset_name}: log10 requires strictly positive MIC values")
        target = np.log10(target_original)
    else:
        target = target_original.copy()
    target.name = "target_model_scale"

    return {
        "name": dataset_name,
        "path": path,
        "X": descriptors,
        "y": target,
        "target_original": target_original,
        "ids": ids.reset_index(drop=True),
        "source_pandas_rows": source_pandas_rows,
        "source_csv_lines": source_csv_lines,
        "n_rows_source": n_rows_source,
        "n_rows_excluded": int(missing_target.sum()) if config["missing_target_policy"] == "exclude" else 0,
        "id_source": id_source,
        "target_source": str(target_col),
    }


def arff_quote(text: Any) -> str:
    clean = str(text).replace("\\", "\\\\").replace("'", "\\'")
    clean = clean.replace("\r", " ").replace("\n", " ")
    return f"'{clean}'"


def format_arff_number(value: Any) -> str:
    if pd.isna(value):
        return "?"
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"WEKA cannot receive an infinite value: {numeric}")
    return format(numeric, ".17g")


def write_numeric_arff(path: Path, X: pd.DataFrame, y: pd.Series, relation: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(X) != len(y):
        raise ValueError("X and y have different lengths")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"@RELATION {arff_quote(relation)}\n\n")
        for column in X.columns:
            handle.write(f"@ATTRIBUTE {arff_quote(column)} NUMERIC\n")
        handle.write("@ATTRIBUTE '__target__' NUMERIC\n\n@DATA\n")
        matrix = X.to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        for row, target_value in zip(matrix, target):
            values = [format_arff_number(value) for value in row]
            values.append(format_arff_number(target_value))
            handle.write(",".join(values) + "\n")


def java_prefix(project_root: Path, config: Dict[str, Any]) -> List[str]:
    executable = os.environ.get("JAVA_EXECUTABLE", "java")
    heap = os.environ.get("JAVA_HEAP", "12g").strip()
    prefix = [executable]
    if heap:
        prefix.append(f"-Xmx{heap}")
    jar_override = os.environ.get("WEKA_JAR")
    jar = Path(jar_override).expanduser().resolve() if jar_override else Path(project_root) / config["weka"]["jar"]
    return prefix + ["-cp", str(jar.resolve())]


def build_classifier_command(
    project_root: Path,
    config: Dict[str, Any],
    train_arff: Path,
    test_arff: Optional[Path] = None,
    prediction_csv: Optional[Path] = None,
    model_path: Optional[Path] = None,
    rf_seed: int = 1,
    no_cv: bool = False,
) -> List[str]:
    command = java_prefix(project_root, config) + [
        "weka.classifiers.meta.AttributeSelectedClassifier",
        "-t",
        str(train_arff),
        "-c",
        "last",
    ]
    if test_arff is not None:
        command += ["-T", str(test_arff)]
    if no_cv:
        command += ["-no-cv"]
    if prediction_csv is not None:
        prediction_spec = (
            "weka.classifiers.evaluation.output.prediction.CSV "
            f'-file "{prediction_csv}" -decimals 16 -suppress'
        )
        command += ["-classifications", prediction_spec]
    if model_path is not None:
        command += ["-d", str(model_path)]
    command += [
        "-E",
        config["weka"]["cfs_spec"],
        "-S",
        config["weka"]["best_first_spec"],
        "-W",
        config["weka"]["random_forest_class"],
        "--",
        "-I",
        str(config["n_trees"]),
        "-S",
        str(rf_seed),
    ]
    return command


def selection_command(
    project_root: Path, config: Dict[str, Any], train_arff: Path
) -> List[str]:
    return java_prefix(project_root, config) + [
        "weka.attributeSelection.CfsSubsetEval",
        "-i",
        str(train_arff),
        "-c",
        "last",
        "-s",
        config["weka"]["best_first_spec"],
    ]


def run_command_to_log(command: Sequence[str], log_path: Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [str(item) for item in command],
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            shell=False,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        rendered = subprocess.list2cmdline([str(item) for item in command])
        raise RuntimeError(
            f"Command exited with code {completed.returncode}:\n{rendered}\n\n{tail}"
        )


def parse_selected_attributes(output: str, feature_names: Sequence[str]) -> List[str]:
    match = re.search(
        r"Selected\s+attributes\s*:\s*([0-9,\-\s]+?)\s*:\s*\d+",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError("'Selected attributes' was not found in the WEKA output")
    indices: List[int] = []
    for token in match.group(1).replace("\n", "").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = map(int, token.split("-", 1))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))
    selected = [feature_names[index - 1] for index in indices if 1 <= index <= len(feature_names)]
    if not selected:
        raise ValueError("WEKA did not return any selected descriptor")
    return selected


def parse_weka_numeric(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("*", "")
    if text in {"", "?"}:
        return np.nan
    if ":" in text and re.fullmatch(r"\d+", text.split(":", 1)[0]):
        text = text.split(":", 1)[1]
    return float(text)


def read_weka_predictions(path: Path, expected_n: int) -> Tuple[Optional[np.ndarray], np.ndarray]:
    predictions = pd.read_csv(path)
    normalized = {str(column).strip().casefold(): column for column in predictions.columns}
    predicted_col = next(
        (original for key, original in normalized.items() if key.startswith("predicted")), None
    )
    actual_col = next(
        (original for key, original in normalized.items() if key.startswith("actual")), None
    )
    if predicted_col is None:
        raise KeyError(f"No predicted column was found in {path}: {list(predictions.columns)}")
    if len(predictions) != expected_n:
        raise ValueError(f"WEKA returned {len(predictions)} predictions; expected {expected_n}")
    predicted = predictions[predicted_col].map(parse_weka_numeric).to_numpy(dtype=float)
    actual = None if actual_col is None else predictions[actual_col].map(parse_weka_numeric).to_numpy(dtype=float)
    if np.isnan(predicted).any():
        raise ValueError(f"Missing predictions were found in {path}")
    return actual, predicted


def regression_metrics(
    actual: Iterable[float], predicted: Iterable[float], target_transform: str
) -> Dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    actual_array, predicted_array = actual_array[valid], predicted_array[valid]
    if len(actual_array) < 2:
        raise ValueError("At least two valid pairs are required")
    pearson = (
        pearsonr(actual_array, predicted_array)[0]
        if np.std(actual_array) > 0 and np.std(predicted_array) > 0
        else np.nan
    )
    spearman = (
        spearmanr(actual_array, predicted_array)[0]
        if np.std(actual_array) > 0 and np.std(predicted_array) > 0
        else np.nan
    )
    denominator_ae = np.sum(np.abs(actual_array - np.mean(actual_array)))
    denominator_se = np.sum((actual_array - np.mean(actual_array)) ** 2)
    if target_transform == "log10":
        actual_mic = 10 ** actual_array
        predicted_mic = 10 ** predicted_array
    else:
        actual_mic = actual_array
        predicted_mic = predicted_array
    positive = (actual_mic > 0) & (predicted_mic > 0)
    twofold_error = np.full(len(actual_mic), np.inf, dtype=float)
    twofold_error[positive] = np.abs(np.log2(predicted_mic[positive] / actual_mic[positive]))
    return {
        "n": int(len(actual_array)),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(math.sqrt(mean_squared_error(actual_array, predicted_array))),
        "r2": float(r2_score(actual_array, predicted_array)),
        "rae_percent": float(100 * np.sum(np.abs(actual_array - predicted_array)) / denominator_ae)
        if denominator_ae
        else np.nan,
        "rrse_percent": float(
            100 * math.sqrt(np.sum((actual_array - predicted_array) ** 2) / denominator_se)
        )
        if denominator_se
        else np.nan,
        "within_1_twofold_percent": float(100 * np.mean(twofold_error <= 1)),
        "within_2_twofold_percent": float(100 * np.mean(twofold_error <= 2)),
        "positive_prediction_percent": float(100 * np.mean(positive)),
    }


def validate_complete_fold(project_root: Path, spec: TaskSpec, run_signature: str) -> bool:
    directory = fold_dir(project_root, spec)
    complete_path = directory / "complete.json"
    prediction_path = directory / "oof_predictions.csv"
    selection_path = directory / "selected_features.csv"
    if not (complete_path.is_file() and prediction_path.is_file() and selection_path.is_file()):
        return False
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(prediction_path)
        selection = pd.read_csv(selection_path)
    except Exception:
        return False
    valid_metadata = (
        complete.get("run_signature") == run_signature
        and int(complete.get("task_id", -1)) == spec.task_id
        and complete.get("dataset") == spec.dataset_name
        and int(complete.get("seed", -1)) == spec.seed
        and int(complete.get("fold", -1)) == spec.fold
    )
    valid_predictions = REQUIRED_PREDICTION_COLUMNS.issubset(predictions.columns) and len(predictions) > 0
    valid_selection = {"dataset", "seed", "fold", "feature"}.issubset(selection.columns) and len(selection) > 0
    return bool(valid_metadata and valid_predictions and valid_selection)
