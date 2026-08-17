#!/usr/bin/env python3
"""Validate the inferred historical pool rule before any audit fold can run."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import atomic_to_csv, atomic_write_json, load_config, project_root_from_file
from leakage_common import (
    audit_root,
    global_selection_dir,
    global_selection_is_complete,
    load_audit_config,
    load_audit_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument(
        "--accept-range-mismatch",
        action="store_true",
        help="Record an explicit override after manually reviewing the generated pools.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    project_config = load_config(root)
    audit_config = load_audit_config(root)
    manifest = load_audit_manifest(root, audit_config)
    signature = str(manifest["run_signature"])
    report_rows = []
    failures = []

    for dataset in project_config["datasets"]:
        name = str(dataset["name"])
        if not global_selection_is_complete(root, audit_config, name, signature):
            failures.append(f"{name}: global selection is incomplete or stale")
            continue
        directory = global_selection_dir(root, audit_config, name)
        summary = pd.read_csv(directory / "pool_summary.csv").sort_values("pool_label")
        labels = summary["pool_label"].astype(int).tolist()
        expected_labels = [int(value) for value in audit_config["pool_labels"]]
        if labels != expected_labels:
            failures.append(f"{name}: pool labels are incomplete or out of order")
            continue
        sizes = summary["n_features"].astype(int).tolist()
        if any(right > left for left, right in zip(sizes, sizes[1:])):
            failures.append(f"{name}: pools are not monotonically nested")

        species = name.split("_", 1)[0]
        ranges = audit_config["historical_pool_size_ranges"][species]
        for label in (0, 90):
            observed = int(summary.loc[summary["pool_label"] == label, "n_features"].iloc[0])
            lower, upper = [int(value) for value in ranges[f"pool_{label}"]]
            passed = lower <= observed <= upper
            report_rows.append(
                {
                    "dataset": name,
                    "species": species,
                    "pool_label": label,
                    "observed_features": observed,
                    "historical_min": lower,
                    "historical_max": upper,
                    "within_historical_range": passed,
                }
            )
            if not passed:
                failures.append(
                    f"{name} pool{label}: observed {observed}, expected {lower}-{upper}"
                )

        configured_label = int(audit_config["historical_pool_labels"][name])
        configured = pd.read_csv(directory / "configured_pool_features.csv")
        expected_size = int(
            summary.loc[summary["pool_label"] == configured_label, "n_features"].iloc[0]
        )
        if len(configured) != expected_size or configured["feature"].duplicated().any():
            failures.append(f"{name}: configured pool file is inconsistent")

    output_dir = audit_root(root, audit_config) / "manifests"
    report = pd.DataFrame(report_rows)
    atomic_to_csv(report, output_dir / "global_pool_range_validation.csv")
    if failures and not args.accept_range_mismatch:
        raise RuntimeError(
            "Global pool validation failed; the full fold array was not authorized:\n- "
            + "\n- ".join(failures)
            + "\nInspect the WEKA logs and pool files. Only after methodological review, "
            "rerun with --accept-range-mismatch to record an explicit override."
        )

    marker = {
        "validated_at": datetime.now().astimezone().isoformat(),
        "run_signature": signature,
        "state": "approved",
        "automatic_historical_range_check_passed": not failures,
        "range_mismatch_override": bool(failures and args.accept_range_mismatch),
        "failures_overridden": failures if args.accept_range_mismatch else [],
        "pool_rule": audit_config["pool_rule"],
    }
    atomic_write_json(marker, output_dir / "GLOBAL_POOLS_APPROVED.json")
    print(report.to_string(index=False), flush=True)
    print("GLOBAL POOLS APPROVED", flush=True)


if __name__ == "__main__":
    main()
