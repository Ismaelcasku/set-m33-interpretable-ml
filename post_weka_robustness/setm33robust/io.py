from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_METADATA = {
    "ID", "Id", "predicted", "sequence", "Sequence", "SEQUENCE", "MIC",
    "mic_pred", "MIC_pred", "modularity_class", "Modularity_Class",
    "pagerank", "pageranks", "PageRank", "cationic_charge",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def completed_output(directory: Path, fingerprint: str, required: list[str]) -> bool:
    return completed_files(
        directory / "run.json",
        fingerprint,
        [directory / name for name in required],
    )


def _manifest_key(marker: Path, output: Path) -> str:
    return os.path.relpath(output.resolve(), marker.parent.resolve())


def completed_files(marker: Path, fingerprint: str, required: list[Path]) -> bool:
    if not marker.exists():
        return False
    try:
        metadata = read_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("status") != "completed" or metadata.get("fingerprint") != fingerprint:
        return False
    manifest = metadata.get("outputs")
    if not isinstance(manifest, dict):
        return False
    for path in required:
        key = _manifest_key(marker, path)
        expected = manifest.get(key)
        if not isinstance(expected, dict) or not path.is_file():
            return False
        size = path.stat().st_size
        if size <= 0 or size != expected.get("size_bytes"):
            return False
        if sha256_file(path) != expected.get("sha256"):
            return False
    return True


def begin_files(marker: Path, fingerprint: str, metadata: dict[str, Any] | None = None) -> None:
    payload = {"status": "running", "fingerprint": fingerprint}
    if metadata:
        payload.update(metadata)
    write_json(marker, payload)


def write_completed_files(
    marker: Path,
    fingerprint: str,
    required: list[Path],
    metadata: dict[str, Any] | None = None,
) -> None:
    outputs = {}
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"cannot complete run: missing/empty output {path}")
        outputs[_manifest_key(marker, path)] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {"status": "completed", "fingerprint": fingerprint, "outputs": outputs}
    if metadata:
        payload.update(metadata)
    write_json(marker, payload)


def begin_output(directory: Path, fingerprint: str, metadata: dict[str, Any] | None = None) -> None:
    begin_files(directory / "run.json", fingerprint, metadata)


def write_completed_output(
    directory: Path,
    fingerprint: str,
    required: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    write_completed_files(
        directory / "run.json",
        fingerprint,
        [directory / name for name in required],
        metadata,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def dataset_path(data_root: Path, task: pd.Series) -> Path:
    return (data_root / str(task["prediction"])).resolve()


def load_prediction(
    data_root: Path,
    task: pd.Series,
    rejected_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[str], int]:
    path = dataset_path(data_root, task)
    if not path.exists():
        raise FileNotFoundError(f"{task['key']}: missing {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    required = {"ID", "predicted", "sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{task['key']}: missing columns {sorted(missing)}")
    frame["predicted"] = pd.to_numeric(frame["predicted"], errors="coerce")
    valid = (
        frame["ID"].notna()
        & frame["predicted"].notna()
        & frame["sequence"].notna()
        & frame["sequence"].astype(str).str.strip().ne("")
    )
    n_rejected = int((~valid).sum())
    if n_rejected and rejected_dir is not None:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        frame.loc[~valid].to_csv(rejected_dir / f"{task['key']}.csv", index=False)
    frame = frame.loc[valid].copy().reset_index(drop=True)
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        duplicated = frame.loc[frame["ID"].duplicated(), "ID"].head().tolist()
        raise ValueError(f"{task['key']}: duplicated peptide IDs, e.g. {duplicated}")
    features = [c for c in frame.columns if c not in PREDICTION_METADATA]
    if not features:
        raise ValueError(f"{task['key']}: no descriptor columns")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    bad = numeric.columns[numeric.isna().any()].tolist()
    if bad:
        raise ValueError(f"{task['key']}: missing/non-numeric values in {bad[:10]}")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{task['key']}: non-finite descriptor values")
    frame[features] = numeric.astype(float)
    return frame, features, n_rejected


def quantile_subset_indices(y: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    n = len(y)
    if max_rows is None or max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=int)
    from sklearn.model_selection import StratifiedShuffleSplit

    ranked = pd.Series(y).rank(method="first")
    bins = pd.qcut(ranked, q=min(10, n), labels=False, duplicates="drop").to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=int(max_rows), random_state=seed)
    selected, _ = next(splitter.split(np.zeros(n), bins))
    return np.sort(selected.astype(int))


def write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def save_shap_reference(
    directory: Path,
    values: np.ndarray,
    features: list[str],
    rows: pd.DataFrame,
    expected_value: float | np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "shap_values.npy", np.asarray(values, dtype=np.float32))
    rows.to_csv(directory / "rows.csv", index=False)
    write_json(directory / "features.json", features)
    write_json(directory / "expected_value.json", np.asarray(expected_value).tolist())


def load_shap_reference(directory: Path) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    values = np.load(directory / "shap_values.npy", mmap_mode="r")
    features = read_json(directory / "features.json")
    rows = pd.read_csv(directory / "rows.csv", dtype={"ID": str})
    if values.shape != (len(rows), len(features)):
        raise ValueError(f"inconsistent SHAP artifact in {directory}")
    return values, features, rows
