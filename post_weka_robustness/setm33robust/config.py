from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "robustness.yml"
TASKS_FILE = PROJECT_ROOT / "config" / "tasks.csv"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    selected = Path(os.environ.get("SET_M33_CONFIG", path or DEFAULT_CONFIG)).expanduser()
    if not selected.is_absolute():
        selected = (PROJECT_ROOT / selected).resolve()
    with selected.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(selected)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def load_tasks() -> pd.DataFrame:
    tasks = pd.read_csv(TASKS_FILE)
    expected = {
        "task_index", "key", "species", "species_label", "charge", "pool",
        "threshold", "prediction",
    }
    missing = expected - set(tasks.columns)
    if missing:
        raise ValueError(f"tasks.csv missing columns: {sorted(missing)}")
    if tasks["task_index"].tolist() != list(range(len(tasks))):
        raise ValueError("task_index must be consecutive and zero-based")
    if tasks["key"].duplicated().any():
        raise ValueError("task keys must be unique")
    return tasks


def task_at(index: int) -> pd.Series:
    tasks = load_tasks()
    if index < 0 or index >= len(tasks):
        raise IndexError(f"task index {index} outside 0..{len(tasks)-1}")
    return tasks.iloc[index]


def resolve_data_root(cfg: dict[str, Any]) -> Path:
    configured = os.environ.get("SET_M33_DATA_ROOT") or cfg.get("data_root")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    root = PROJECT_ROOT
    candidates.extend([
        root / "data" / "Supplementary Data",
        root / "Supplementary Data",
        root.parent / "Supplementary Data",
        root.parent / "data" / "Supplementary Data",
        root.parent / "data",
        root.parent.parent / "Supplementary Data",
    ])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "Test" / "Analogs predictions").is_dir():
            return resolved
    tested = "\n  - ".join(str(x.resolve()) for x in candidates)
    raise FileNotFoundError(
        "Could not find a data root containing Test/Analogs predictions/. "
        f"Set data_root in config/robustness.yml or SET_M33_DATA_ROOT. Tested:\n  - {tested}"
    )


def resolve_results_root(cfg: dict[str, Any]) -> Path:
    raw = os.environ.get("SET_M33_RESULTS_ROOT") or cfg.get("results_root", "results")
    result = Path(raw).expanduser()
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    result.mkdir(parents=True, exist_ok=True)
    return result.resolve()


def umap_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    section = cfg["umap"]
    species = load_tasks()["species"].drop_duplicates().tolist()
    jobs = []
    for i, values in enumerate(itertools.product(
        species,
        section["n_neighbors"],
        section["min_dist"],
    )):
        sp, n_neighbors, min_dist = values
        jobs.append({
            "umap_task_index": i,
            "species": sp,
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
        })
    return jobs


def cpu_count() -> int:
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
