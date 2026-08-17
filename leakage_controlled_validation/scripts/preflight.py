#!/usr/bin/env python3
"""Audit inputs and software, then create the immutable task manifest."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from common import (
    atomic_to_csv,
    atomic_write_json,
    canonical_sha256,
    dataset_path,
    file_sha256,
    find_column,
    find_invalid_target_rows,
    load_config,
    manifest_path,
    project_root_from_file,
    target_masks,
    task_from_id,
    total_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = (completed.stdout or "") + ("\n" if completed.stdout else "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{output[-5000:]}")
    return output.strip()


def parse_weka_version(output: str) -> str:
    match = re.search(r"(?m)^\s*(\d+\.\d+\.\d+)\s*$", output)
    if not match:
        raise ValueError(f"Could not parse the WEKA version from:\n{output[-3000:]}")
    return match.group(1)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config = load_config(root, args.config)
    for relative in ("input", "logs", "results/folds", "results/manifests", "results/summary", "software"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    jar_override = __import__("os").environ.get("WEKA_JAR")
    jar = Path(jar_override).expanduser().resolve() if jar_override else (root / config["weka"]["jar"]).resolve()
    if not jar.is_file():
        raise FileNotFoundError(
            f"WEKA JAR was not found at {jar}. Copy WEKA 3.8.6 to software/weka.jar "
            "or export WEKA_JAR=/absolute/path/to/weka.jar."
        )

    java_executable = __import__("os").environ.get("JAVA_EXECUTABLE", "java")
    java_version = command_output([java_executable, "-version"])
    weka_raw = command_output([java_executable, "-cp", str(jar), "weka.core.Version"])
    weka_version = parse_weka_version(weka_raw)
    expected_weka = str(config["weka"]["expected_version"])
    if weka_version != expected_weka:
        raise RuntimeError(f"Expected WEKA {expected_weka}, but {weka_version} was found in {jar}")

    print(f"Project: {root}", flush=True)
    print(f"Java: {java_version.splitlines()[0]}", flush=True)
    print(f"WEKA: {weka_version} ({jar})", flush=True)

    input_rows = []
    target_issue_frames = []
    for position, dataset in enumerate(config["datasets"], start=1):
        name = str(dataset["name"])
        path = dataset_path(root, dataset)
        print(f"[{position}/{len(config['datasets'])}] auditing {name}: {path.name}", flush=True)
        if not path.is_file():
            raise FileNotFoundError(f"Missing input for {name}: {path}")
        header = pd.read_csv(path, sep=config["csv_separator"], nrows=0)
        target_col = find_column(header, config["target_column"], required=True)
        id_col = find_column(header, config["id_column"], required=False)
        usecols = [target_col] + ([id_col] if id_col is not None else [])
        target_frame = pd.read_csv(path, sep=config["csv_separator"], usecols=usecols, low_memory=False)
        parsed, missing, non_numeric = target_masks(target_frame[target_col])
        target_issue_frames.append(find_invalid_target_rows(name, path, config))
        included = parsed.loc[~missing]
        excluded_columns = 1 + (1 if id_col is not None else 0)
        stat = path.stat()
        input_rows.append(
            {
                "dataset": name,
                "file": str(dataset["file"]),
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "n_rows_source": int(len(target_frame)),
                "n_rows_excluded_missing_mic": int(missing.sum()),
                "n_rows_model": int((~missing).sum()),
                "n_non_numeric_mic": int(non_numeric.sum()),
                "n_descriptors": int(len(header.columns) - excluded_columns),
                "id_source": str(id_col) if id_col is not None else "generated_from_source_row_number",
                "target_source": str(target_col),
                "target_transform": config["target_transform"],
                "mic_min_original": float(included.min()),
                "mic_median_original": float(included.median()),
                "mic_max_original": float(included.max()),
            }
        )

    invalid_targets = pd.concat(target_issue_frames, ignore_index=True)
    manifests_dir = root / "results" / "manifests"
    atomic_to_csv(invalid_targets, manifests_dir / "invalid_target_values.csv")
    non_numeric_targets = invalid_targets.query("problem == 'non_numeric'").copy()
    if not non_numeric_targets.empty:
        print(non_numeric_targets.to_string(index=False))
        raise ValueError(
            f"Found {len(non_numeric_targets)} non-numeric MIC values. They were not converted or excluded."
        )
    excluded_missing = invalid_targets.query("problem == 'missing'").copy()
    atomic_to_csv(excluded_missing, manifests_dir / "excluded_missing_targets.csv")
    if not excluded_missing.empty and config["missing_target_policy"] == "error":
        raise ValueError(f"Found {len(excluded_missing)} missing MIC values and policy='error'.")

    signature_payload = {
        "config": config,
        "weka_version": weka_version,
        "inputs": {row["dataset"]: row["sha256"] for row in input_rows},
    }
    run_signature = canonical_sha256(signature_payload)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "method": (
            "Repeated explicit five-fold CV; missing MIC excluded before splitting; "
            "CfsSubsetEval + BestFirst fitted only on each training fold; "
            "RandomForest predicts the intact held-out fold"
        ),
        "run_signature": run_signature,
        "python": sys.version,
        "platform": platform.platform(),
        "java_version": java_version,
        "weka_version": weka_version,
        "weka_jar": str(jar),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": config,
        "target_audit": {
            "missing_occurrences_excluded": int(len(excluded_missing)),
            "missing_unique_ids": int(excluded_missing["sample_id"].nunique()) if not excluded_missing.empty else 0,
            "non_numeric_occurrences": int(len(non_numeric_targets)),
            "original_csv_files_modified": False,
        },
        "inputs": input_rows,
    }

    output_manifest = manifest_path(root)
    if output_manifest.is_file():
        previous = json.loads(output_manifest.read_text(encoding="utf-8"))
        if previous.get("run_signature") != run_signature:
            raise RuntimeError(
                "Existing results belong to different inputs or configuration. "
                "Move them to a separate directory before starting this run."
            )
        manifest["created_at"] = previous.get("created_at", manifest["created_at"])

    atomic_write_json(manifest, output_manifest)
    atomic_to_csv(pd.DataFrame(input_rows), manifests_dir / "input_audit.csv")
    tasks = []
    for task_id in range(total_tasks(config)):
        spec = task_from_id(config, task_id)
        tasks.append(
            {
                "task_id": spec.task_id,
                "dataset_index": spec.dataset_index,
                "dataset": spec.dataset_name,
                "file": spec.dataset_file,
                "seed_index": spec.seed_index,
                "seed": spec.seed,
                "fold": spec.fold,
            }
        )
    atomic_to_csv(pd.DataFrame(tasks), manifests_dir / "task_manifest.csv")
    print(
        f"PREFLIGHT COMPLETE: {len(input_rows)} datasets, {len(tasks)} tasks, "
        f"{len(excluded_missing)} missing-MIC occurrences excluded "
        f"({manifest['target_audit']['missing_unique_ids']} unique IDs).",
        flush=True,
    )
    print(f"Run signature: {run_signature}", flush=True)


if __name__ == "__main__":
    main()
