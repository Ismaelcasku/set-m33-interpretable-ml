#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from setm33robust.config import load_config, load_tasks, resolve_data_root, resolve_results_root, umap_jobs
from setm33robust.io import dataset_path, load_prediction, sha256_file, write_json


EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.2",
    "networkx": "3.3",
    "shap": "0.46.0",
    "umap-learn": "0.5.6",
    "hdbscan": "0.8.40",
    "numba": "0.60.0",
    "llvmlite": "0.43.0",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["umap"]["metric"] != "euclidean":
        raise ValueError("historical UMAP metric must be euclidean")
    if cfg["umap"]["descriptor_group_k"] != {
        "ecoli": 6, "paeruginosa": 6, "kpneumoniae": 3,
    }:
        raise ValueError("historical descriptor-group K must be 6/6/3")
    tasks = load_tasks()
    data_root = resolve_data_root(cfg)
    results = resolve_results_root(cfg)
    rows = []
    manifest_path = ROOT / "data" / "input_manifest.sha256"
    expected_hashes = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split("  ", 1)
        prefix = "data/Supplementary Data/"
        expected_hashes[relative[len(prefix):] if relative.startswith(prefix) else relative] = checksum
    feature_sets: dict[str, list[set[str]]] = {}
    for _, task in tasks.iterrows():
        frame, features, rejected = load_prediction(data_root, task, results / "preflight" / "rejected_rows")
        constant = int((frame[features].std(axis=0, ddof=0) == 0).sum())
        if constant:
            raise ValueError(f"{task['key']}: {constant} constant descriptors prevent HSPN z-scoring")
        checksum = sha256_file(dataset_path(data_root, task))
        expected = expected_hashes.get(str(task["prediction"]))
        if expected is None or checksum != expected:
            raise ValueError(f"{task['key']}: SHA-256 does not match bundled canonical input")
        rows.append({
            "task_index": int(task["task_index"]),
            "key": task["key"],
            "species": task["species"],
            "charge": int(task["charge"]),
            "pool": int(task["pool"]),
            "selected_threshold": float(task["threshold"]),
            "path": str(dataset_path(data_root, task)),
            "rows_in_file": len(frame) + rejected,
            "valid_rows": len(frame),
            "rejected_rows": rejected,
            "n_descriptors": len(features),
            "constant_descriptors": constant,
            "sha256": checksum,
        })
        feature_sets.setdefault(task["species"], []).append(set(features))
    audit = pd.DataFrame(rows)
    out = results / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "dataset_audit.csv", index=False)
    align = []
    for species, sets in feature_sets.items():
        align.append({
            "species": species,
            "n_charge_sets": len(sets),
            "intersection_descriptors": len(set.intersection(*sets)),
            "union_descriptors": len(set.union(*sets)),
        })
    pd.DataFrame(align).to_csv(out / "descriptor_alignment_audit.csv", index=False)
    packages = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    mismatches = {
        name: {"expected": expected, "observed": packages[name]}
        for name, expected in EXPECTED_VERSIONS.items()
        if packages[name] != expected
    }
    python_mismatch = sys.version_info[:2] != (3, 11)
    if (mismatches or python_mismatch) and os.environ.get("SET_M33_ALLOW_VERSION_MISMATCH") != "1":
        raise RuntimeError(
            "Pinned environment mismatch. Recreate the environment before the full run. "
            f"python={platform.python_version()}, package_mismatches={mismatches}"
        )
    write_json(out / "preflight.json", {
        "status": "passed",
        "data_root": str(data_root),
        "results_root": str(results),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "version_mismatches_allowed": bool(mismatches or python_mismatch),
        "n_tasks": len(tasks),
        "n_umap_array_tasks": len(umap_jobs(cfg)),
    })
    print(audit.to_string(index=False))
    print("\nDescriptor alignment:")
    print(pd.DataFrame(align).to_string(index=False))
    print(
        f"\nPreflight PASSED: {len(tasks)} datasets; "
        f"3 historical UMAP tasks; {len(umap_jobs(cfg))} robustness UMAP tasks"
    )


if __name__ == "__main__":
    main()
