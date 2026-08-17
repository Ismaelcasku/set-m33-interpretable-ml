from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def make_surrogate(section: dict[str, Any], seed: int, n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(section["n_estimators"]),
        max_depth=None if section.get("max_depth") is None else int(section["max_depth"]),
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )


def tree_shap(model: RandomForestRegressor, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import shap

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )
    result = explainer.shap_values(values, check_additivity=False)
    if hasattr(result, "values"):
        result = result.values
    shap_values = np.asarray(result, dtype=float)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.ndim != 2:
        raise ValueError(f"unexpected SHAP shape {shap_values.shape}")
    return shap_values, np.asarray(explainer.expected_value)
