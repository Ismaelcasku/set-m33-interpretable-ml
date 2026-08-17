#!/usr/bin/env python3
"""Reconstruct SET-M33 physicochemical similarity networks and rank candidates.

This module is extracted from the network/candidate-selection section of the
original network-analysis workflow. It intentionally does not implement
SHAP, UMAP, or the leakage audit; those final analyses live in dedicated
modules at the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

PREDICTION_METADATA = {"ID", "Id", "MIC", "sequence", "SEQUENCE", "Class", "predicted"}
TRAIN_METADATA = PREDICTION_METADATA
DEFAULT_NETWORK_SEED = 42
THRESHOLD_SWEEP = np.round(np.arange(0.05, 1.00, 0.05), 2)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("SET_M33_DATA_ROOT", ".")),
        help="Directory containing Training/ and Test/ from Supplementary Data.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=root / "config" / "tasks.csv",
        help="Canonical task registry.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "results",
        help="Output directory.",
    )
    parser.add_argument(
        "--task",
        default="all",
        help="One task key from config/tasks.csv or 'all'.",
    )
    parser.add_argument(
        "--network-seed", type=int, default=DEFAULT_NETWORK_SEED,
        help="Seed for deterministic NetworkX Louvain reconstruction.",
    )
    parser.add_argument(
        "--full-topology-sweep", action="store_true",
        help="Also calculate components, communities, modularity and clustering at every threshold.",
    )
    return parser.parse_args()


def descriptor_columns(frame: pd.DataFrame, metadata: set[str]) -> list[str]:
    return [str(column) for column in frame.columns if str(column) not in metadata]


def numeric_frame(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    converted = frame[columns].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        bad = converted.columns[converted.isna().any()].tolist()[:10]
        raise ValueError(f"{label}: missing or non-numeric descriptor values in {bad}")
    values = converted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: infinite descriptor values found")
    return converted.astype(float)


def load_valid_analogs(data_root: Path, task: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = (data_root / str(task.prediction)).resolve()
    frame = pd.read_csv(path, low_memory=False)
    required = {"ID", "predicted", "sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{task.key}: missing analog metadata columns {sorted(missing)}")
    predicted = pd.to_numeric(frame["predicted"], errors="coerce")
    invalid = (
        frame["ID"].isna()
        | frame["ID"].astype("string").str.strip().eq("")
        | frame["sequence"].isna()
        | frame["sequence"].astype("string").str.strip().eq("")
        | predicted.isna()
        | ~np.isfinite(predicted.fillna(np.nan))
        | predicted.le(0)
    )
    excluded = frame.loc[invalid, ["ID", "predicted", "sequence"]].copy()
    excluded.insert(0, "source_csv_line", excluded.index.to_numpy(dtype=int) + 2)
    valid = frame.loc[~invalid].copy().reset_index(drop=True)
    valid["predicted"] = pd.to_numeric(valid["predicted"], errors="raise").astype(float)
    if valid["ID"].duplicated().any():
        raise ValueError(f"{task.key}: duplicate analog IDs in the valid cohort")
    return valid, excluded


def validate_descriptor_identity(data_root: Path, task: pd.Series) -> list[str]:
    training_path = (data_root / str(task.training_pool)).resolve()
    prediction_path = (data_root / str(task.prediction)).resolve()
    if not training_path.is_file() or not prediction_path.is_file():
        missing = [str(p) for p in (training_path, prediction_path) if not p.is_file()]
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    train_header = pd.read_csv(training_path, nrows=0)
    pred_header = pd.read_csv(prediction_path, nrows=0)
    train_features = descriptor_columns(train_header, TRAIN_METADATA)
    pred_features = descriptor_columns(pred_header, PREDICTION_METADATA)
    if set(train_features) != set(pred_features):
        raise ValueError(
            f"{task.key}: final training and prediction descriptor sets differ; "
            f"training-only={sorted(set(train_features)-set(pred_features))[:10]}, "
            f"prediction-only={sorted(set(pred_features)-set(train_features))[:10]}"
        )
    return pred_features


def network_input(data_root: Path, task: pd.Series) -> tuple[pd.DataFrame, list[str], np.ndarray, pd.DataFrame]:
    features = validate_descriptor_identity(data_root, task)
    frame, excluded = load_valid_analogs(data_root, task)
    X = numeric_frame(frame, features, str(task.key))
    nonconstant = X.columns[X.std(axis=0, ddof=0) > 0].tolist()
    if not nonconstant:
        raise ValueError(f"{task.key}: all descriptors are constant")
    if len(nonconstant) != len(features):
        warnings.warn(
            f"{task.key}: dropping {len(features) - len(nonconstant)} constant descriptors",
            stacklevel=2,
        )
    scaled = StandardScaler().fit_transform(X[nonconstant]).astype(np.float32)
    return frame, nonconstant, scaled, excluded


def similarity_upper_triangle(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    similarity = cosine_similarity(X_scaled, dense_output=True).astype(np.float32)
    np.fill_diagonal(similarity, np.nan)
    i_idx, j_idx = np.triu_indices_from(similarity, k=1)
    return i_idx, j_idx, similarity[i_idx, j_idx]


def threshold_sweep_table(
    n_nodes: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    similarities: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        keep = similarities >= threshold
        n_edges = int(keep.sum())
        if n_edges:
            active = np.unique(np.concatenate([i_idx[keep], j_idx[keep]]))
            n_nonisolated = int(len(active))
        else:
            n_nonisolated = 0
        rows.append(
            {
                "threshold": float(threshold),
                "n_nodes_all": int(n_nodes),
                "n_nodes_nonisolated": n_nonisolated,
                "n_isolates": int(n_nodes - n_nonisolated),
                "n_edges": n_edges,
                "density_all_nodes": 0.0
                if n_nodes < 2
                else 2 * n_edges / (n_nodes * (n_nodes - 1)),
            }
        )
    return pd.DataFrame(rows)


def build_graph_at_threshold(
    frame: pd.DataFrame,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    similarities: np.ndarray,
    threshold: float,
) -> tuple[nx.Graph, np.ndarray]:
    graph = nx.Graph()
    ids = frame["ID"].astype(str).to_numpy()
    graph.add_nodes_from(ids)
    keep = similarities >= threshold
    graph.add_weighted_edges_from(
        (ids[i], ids[j], float(value))
        for i, j, value in zip(i_idx[keep], j_idx[keep], similarities[keep])
    )
    return graph, keep


def deterministic_louvain_labels(graph: nx.Graph, seed: int = DEFAULT_NETWORK_SEED) -> dict[str, int]:
    if graph.number_of_edges() == 0:
        return {str(node): index for index, node in enumerate(sorted(graph.nodes()))}
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=1.0, seed=seed
    )
    ordered = sorted(
        (sorted(map(str, community)) for community in communities), key=lambda x: x[0]
    )
    return {
        node: label
        for label, community in enumerate(ordered)
        for node in community
    }


def select_candidates(nodes: pd.DataFrame) -> pd.DataFrame:
    """Apply the study module/PageRank candidate-prioritization rule."""
    if nodes.empty:
        return nodes.copy()
    module_mean = (
        nodes.groupby("modularity_class", as_index=True)["predicted"].mean().sort_values()
    )
    n_modules = max(1, math.ceil(len(module_mean) / 2))
    retained = nodes.loc[
        nodes["modularity_class"].isin(module_mean.index[:n_modules])
    ].copy()
    selected: list[pd.DataFrame] = []
    for _, group in retained.groupby("modularity_class", sort=True):
        n_keep = max(1, math.ceil(len(group) / 2))
        selected.append(
            group.sort_values(["pagerank", "predicted"], ascending=[False, True]).head(n_keep)
        )
    return pd.concat(selected, ignore_index=True).sort_values("predicted", ascending=True)


def topology_metrics(graph: nx.Graph, labels: dict[str, int]) -> dict[str, float | int]:
    communities = [
        {node for node, label in labels.items() if label == value}
        for value in sorted(set(labels.values()))
    ]
    modularity = (
        float(nx.community.modularity(graph, communities, weight="weight"))
        if graph.number_of_edges()
        else float("nan")
    )
    return {
        "n_components": nx.number_connected_components(graph),
        "n_communities": len(communities),
        "modularity": modularity,
        "mean_clustering": float(nx.average_clustering(graph, weight="weight")),
    }


def run_network_task(
    data_root: Path,
    output_root: Path,
    task: pd.Series,
    network_seed: int,
    full_topology_sweep: bool,
) -> dict[str, float | int | str]:
    frame, features, X_scaled, excluded = network_input(data_root, task)
    i_idx, j_idx, similarities = similarity_upper_triangle(X_scaled)
    sweep = threshold_sweep_table(
        len(frame), i_idx, j_idx, similarities, THRESHOLD_SWEEP
    )

    if full_topology_sweep:
        rows = []
        for threshold in THRESHOLD_SWEEP:
            graph, _ = build_graph_at_threshold(
                frame, i_idx, j_idx, similarities, float(threshold)
            )
            labels = deterministic_louvain_labels(graph, network_seed)
            rows.append({"threshold": float(threshold), **topology_metrics(graph, labels)})
        sweep = sweep.merge(pd.DataFrame(rows), on="threshold", how="left", validate="one_to_one")

    out = output_root / str(task.key)
    out.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(out / "excluded_rows.csv", index=False)
    sweep.to_csv(out / "threshold_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sweep["threshold"], sweep["density_all_nodes"], linewidth=1.8)
    axes[1].plot(sweep["threshold"], sweep["n_nodes_nonisolated"], linewidth=1.8)
    for axis in axes:
        axis.axvline(float(task.threshold), linestyle="--", linewidth=1.5)
        axis.set_xlabel("Cosine-similarity threshold")
    axes[0].set_ylabel("Network density")
    axes[1].set_ylabel("Non-isolated nodes")
    fig.suptitle(f"{task.species_label}, charge +{int(task.charge)}, pool {int(task.pool)}")
    fig.tight_layout()
    fig.savefig(out / "threshold_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)

    graph, keep = build_graph_at_threshold(
        frame, i_idx, j_idx, similarities, float(task.threshold)
    )
    communities = deterministic_louvain_labels(graph, network_seed)
    pagerank = (
        nx.pagerank(graph, weight="weight")
        if graph.number_of_edges()
        else {node: 0.0 for node in graph}
    )
    component: dict[str, int] = {}
    ordered_components = sorted(
        nx.connected_components(graph), key=lambda x: (-len(x), min(map(str, x)))
    )
    for component_id, members in enumerate(ordered_components):
        for node in members:
            component[str(node)] = component_id

    nodes = frame[["ID", "sequence", "predicted"]].copy()
    nodes["ID"] = nodes["ID"].astype(str)
    nodes["modularity_class"] = nodes["ID"].map(communities).astype(int)
    nodes["pagerank"] = nodes["ID"].map(pagerank).astype(float)
    nodes["degree"] = nodes["ID"].map(dict(graph.degree())).astype(int)
    nodes["weighted_degree"] = nodes["ID"].map(dict(graph.degree(weight="weight"))).astype(float)
    nodes["component"] = nodes["ID"].map(component).astype(int)
    nodes["is_isolate"] = nodes["degree"].eq(0)

    ids = frame["ID"].astype(str).to_numpy()
    edges = pd.DataFrame(
        {
            "Source": ids[i_idx[keep]],
            "Target": ids[j_idx[keep]],
            "Similarity": similarities[keep].astype(float),
        }
    )
    candidates = select_candidates(nodes)
    nodes.to_csv(out / "nodes_selected_threshold.csv", index=False)
    edges.to_csv(out / "edges_selected_threshold.csv", index=False)
    candidates.to_csv(out / "candidates_ranked.csv", index=False)

    metrics: dict[str, float | int | str] = {
        "key": str(task.key),
        "threshold": float(task.threshold),
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_isolates": int(sum(1 for _ in nx.isolates(graph))),
        "density": float(nx.density(graph)),
        "n_candidates": len(candidates),
        "n_descriptors": len(features),
        **topology_metrics(graph, communities),
    }
    (out / "selected_network_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def load_tasks(path: Path) -> pd.DataFrame:
    tasks = pd.read_csv(path)
    required = {
        "key", "species", "species_label", "charge", "pool", "threshold",
        "training_pool", "prediction",
    }
    missing = required - set(tasks.columns)
    if missing:
        raise ValueError(f"Task registry lacks columns: {sorted(missing)}")
    if len(tasks) != 11:
        raise ValueError(f"Expected 11 canonical target/charge tasks, found {len(tasks)}")
    kp = tasks.loc[tasks["species"].eq("kpneumoniae")].set_index("charge")["pool"].to_dict()
    if kp != {4: 10, 5: 60, 6: 60}:
        raise ValueError(f"Canonical K. pneumoniae pools must be 10/60/60, found {kp}")
    return tasks


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tasks = load_tasks(args.tasks.expanduser().resolve())
    if args.task != "all":
        tasks = tasks.loc[tasks["key"].eq(args.task)]
        if tasks.empty:
            raise KeyError(f"Unknown task key: {args.task}")

    output_root.mkdir(parents=True, exist_ok=True)
    metrics = [
        run_network_task(
            data_root, output_root, task, args.network_seed, args.full_topology_sweep
        )
        for _, task in tasks.iterrows()
    ]
    pd.DataFrame(metrics).to_csv(output_root / "all_selected_network_metrics.csv", index=False)
    print(f"Completed {len(metrics)} network task(s) in {output_root}")


if __name__ == "__main__":
    main()
