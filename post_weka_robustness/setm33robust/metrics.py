from __future__ import annotations

from itertools import combinations
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import NearestNeighbors


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = stats.spearmanr(a, b, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": safe_spearman(y_true, y_pred),
    }


def percentile_ci(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (1 - confidence) / 2
    return float(data.mean()), float(np.quantile(data, alpha)), float(np.quantile(data, 1 - alpha))


def ranks_desc(values: np.ndarray) -> np.ndarray:
    return pd.Series(-np.asarray(values, dtype=float)).rank(method="average").to_numpy()


def jaccard(a: Iterable, b: Iterable) -> float:
    aa, bb = set(a), set(b)
    union = aa | bb
    return 1.0 if not union else len(aa & bb) / len(union)


def pairwise_jaccards(sets: list[set]) -> np.ndarray:
    return np.asarray([jaccard(a, b) for a, b in combinations(sets, 2)], dtype=float)


def knn_indices(values: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """Return the first k non-self neighbors of every fitted observation.

    With ``X=None``, scikit-learn's ``kneighbors`` already excludes each query
    observation from its own neighbor list. No additional column is removed.
    """
    n = len(values)
    k_use = min(int(k), n - 1)
    if k_use < 1:
        raise ValueError("at least two rows are required for kNN metrics")
    model = NearestNeighbors(n_neighbors=k_use, metric=metric, n_jobs=1)
    return model.fit(values).kneighbors(return_distance=False)


def mean_knn_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Mean conventional kNN overlap: ``|N1(i) ∩ N2(i)| / k``."""
    if first.shape != second.shape:
        raise ValueError("neighbor arrays must have identical shapes")
    if first.ndim != 2 or first.shape[1] < 1:
        raise ValueError("neighbor arrays must be non-empty two-dimensional matrices")
    k = first.shape[1]
    return float(np.mean([
        len(set(first[i].tolist()) & set(second[i].tolist())) / k
        for i in range(len(first))
    ]))


def effective_dimension(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - np.mean(values, axis=0, keepdims=True)
    eigenvalues = np.linalg.svd(centered, full_matrices=False, compute_uv=False) ** 2
    denominator = float(np.square(eigenvalues).sum())
    return 0.0 if denominator == 0 else float(eigenvalues.sum() ** 2 / denominator)


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def balanced_subset_indices(labels: np.ndarray, max_per_group: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    groups = np.unique(labels)
    n_each = min(int(max_per_group), *(int(np.sum(labels == group)) for group in groups))
    rng = np.random.default_rng(seed)
    pieces = [rng.choice(np.flatnonzero(labels == group), size=n_each, replace=False) for group in groups]
    return np.sort(np.concatenate(pieces).astype(int))
