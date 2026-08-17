#!/usr/bin/env python3
"""Report validated progress and emit a compact retry array specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import atomic_to_csv, load_config, project_root_from_file
from leakage_common import (
    audit_fold_dir,
    audit_fold_is_complete,
    audit_root,
    audit_task_from_id,
    global_selection_is_complete,
    load_audit_config,
    load_audit_manifest,
    total_audit_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--missing-array", action="store_true")
    parser.add_argument(
        "--branch",
        choices=["legacy_global", "nested_matched"],
        help="With --missing-array, emit branch-local task IDs in the range 0-549.",
    )
    return parser.parse_args()


def compact_ranges(values: list[int]) -> str:
    if not values:
        return ""
    ordered = sorted(set(values))
    chunks = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        chunks.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    chunks.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(chunks)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    project_config = load_config(root)
    audit_config = load_audit_config(root)
    manifest = load_audit_manifest(root, audit_config)
    signature = str(manifest["run_signature"])
    output_root = audit_root(root, audit_config)

    global_rows = []
    for dataset in project_config["datasets"]:
        name = str(dataset["name"])
        complete = global_selection_is_complete(root, audit_config, name, signature)
        global_rows.append({"dataset": name, "complete": complete})
    global_frame = pd.DataFrame(global_rows)

    rows = []
    missing = []
    for task_id in range(total_audit_tasks(project_config, audit_config)):
        spec = audit_task_from_id(project_config, audit_config, task_id)
        complete = audit_fold_is_complete(root, audit_config, spec, signature)
        state = "complete" if complete else "missing"
        if not complete:
            missing.append(task_id)
            status_path = audit_fold_dir(root, audit_config, spec) / "status.json"
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    state = str(status.get("state", "invalid"))
                except Exception:
                    state = "invalid_status"
        rows.append(
            {
                "task_id": task_id,
                "branch": spec.branch,
                "dataset": spec.base.dataset_name,
                "seed": spec.base.seed,
                "fold": spec.base.fold,
                "state": state,
                "validated_complete": complete,
            }
        )
    frame = pd.DataFrame(rows)
    status_dir = output_root / "progress"
    atomic_to_csv(global_frame, status_dir / "global_selection_status.csv")
    atomic_to_csv(frame, status_dir / "fold_task_status.csv")
    array_spec = compact_ranges(missing)
    (status_dir / "missing_array_spec.txt").parent.mkdir(parents=True, exist_ok=True)
    (status_dir / "missing_array_spec.txt").write_text(array_spec + "\n", encoding="utf-8")

    if args.missing_array:
        if args.branch:
            branch_offset = {
                "legacy_global": 0,
                "nested_matched": 550,
            }[args.branch]
            branch_missing = [
                int(row.task_id) - branch_offset
                for row in frame.itertuples()
                if row.branch == args.branch and not bool(row.validated_complete)
            ]
            print(compact_ranges(branch_missing))
        else:
            print(array_spec)
        return

    print(
        f"Global selections: {int(global_frame['complete'].sum())}/{len(global_frame)} complete",
        flush=True,
    )
    summary = frame.groupby(["branch", "dataset"])["validated_complete"].agg(
        ["sum", "count"]
    ).reset_index()
    summary["progress"] = summary["sum"].astype(int).astype(str) + "/" + summary["count"].astype(int).astype(str)
    print(summary[["branch", "dataset", "progress"]].to_string(index=False), flush=True)
    print(
        f"Fold tasks: {len(frame) - len(missing)}/{len(frame)} complete; missing={len(missing)}",
        flush=True,
    )
    if missing:
        print(f"Retry array: {array_spec}", flush=True)


if __name__ == "__main__":
    main()
