#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from setm33robust.config import load_config, resolve_data_root, resolve_results_root, task_at
from setm33robust.io import (
    analysis_fingerprint, begin_output, completed_output, dataset_path,
    load_prediction, sha256_file, write_completed_output,
)
from setm33robust.metrics import jaccard, percentile_ci, safe_spearman


def similarity_triangle(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sd = X.std(axis=0, ddof=0)
    if np.any(sd == 0):
        raise ValueError("constant descriptors prevent historical HSPN z-scoring")
    standardized = (X - X.mean(axis=0)) / sd
    unit = normalize(standardized, norm="l2", axis=1).astype(np.float32, copy=False)
    similarity = unit @ unit.T
    i_idx, j_idx = np.triu_indices(len(X), k=1)
    values = similarity[i_idx, j_idx].astype(np.float32, copy=False)
    del similarity, unit
    return i_idx.astype(np.int32), j_idx.astype(np.int32), values


def graph_at_threshold(n: int, i_idx: np.ndarray, j_idx: np.ndarray, values: np.ndarray, threshold: float) -> nx.Graph:
    keep = values >= threshold
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(
        (int(i), int(j), float(value))
        for i, j, value in zip(i_idx[keep], j_idx[keep], values[keep])
    )
    return graph


def partition_labels(communities: list[set[int]], n: int) -> np.ndarray:
    ordered = sorted((set(c) for c in communities), key=lambda c: min(c))
    labels = np.full(n, -1, dtype=int)
    for label, community in enumerate(ordered):
        labels[list(community)] = label
    if np.any(labels < 0):
        raise ValueError("incomplete Louvain partition")
    return labels


def historical_candidates(labels: np.ndarray, pagerank: np.ndarray, predicted: np.ndarray) -> set[int]:
    table = pd.DataFrame({"row": np.arange(len(labels)), "module": labels, "pagerank": pagerank, "predicted": predicted})
    means = table.groupby("module")["predicted"].mean().sort_values(kind="stable")
    retained_modules = means.iloc[: len(means) // 2].index
    selected: set[int] = set()
    for _, group in table.loc[table["module"].isin(retained_modules)].groupby("module", sort=False):
        n_top = len(group) // 2
        selected.update(group.sort_values("pagerank", ascending=False, kind="stable").iloc[:n_top]["row"].astype(int))
    return selected


def summarize_pairwise(values: list[float], prefix: str) -> dict[str, float]:
    mean, low, high = percentile_ci(values)
    return {f"{prefix}_mean": mean, f"{prefix}_ci95_low": low, f"{prefix}_ci95_high": high, f"{prefix}_min": float(np.min(values)) if values else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--louvain-seeds", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    task = task_at(args.task_index)
    data_root = resolve_data_root(cfg)
    frame, features, _ = load_prediction(data_root, task)
    X = frame[features].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    ids = frame["ID"].to_numpy(dtype=str)
    n_seeds = int(args.louvain_seeds or cfg["hspn"]["louvain_seeds"])
    fingerprint = analysis_fingerprint({
        "stage": "hspn_stability", "key": task["key"],
        "input_sha256": sha256_file(dataset_path(data_root, task)),
        "config": cfg["hspn"], "n_louvain_seeds": n_seeds,
    })
    out = resolve_results_root(cfg) / "hspn" / str(task["key"])
    required = [
        "threshold_sweep.csv", "threshold_sweep.png", "louvain_runs.csv",
        "threshold_stability_summary.csv", "adjacent_threshold_comparison.csv",
        "candidate_selection_frequency.csv.gz", "row_order.csv",
        "louvain_partitions_and_candidates.npz",
    ]
    if completed_output(out, fingerprint, required):
        print(f"skip completed HSPN stability: {task['key']}")
        return
    out.mkdir(parents=True, exist_ok=True)
    begin_output(out, fingerprint, {"key": task["key"]})
    i_idx, j_idx, similarities = similarity_triangle(X)
    sweep_rows = []
    for threshold in cfg["hspn"]["threshold_sweep"]:
        keep = similarities >= float(threshold)
        n_edges = int(keep.sum())
        incident = np.unique(np.concatenate([i_idx[keep], j_idx[keep]])) if n_edges else np.array([], dtype=int)
        n_incident = len(incident)
        sweep_rows.append({
            "key": task["key"], "threshold": float(threshold), "n_nodes_all": len(frame),
            "n_nodes_incident": n_incident, "n_isolates": len(frame) - n_incident,
            "n_edges": n_edges,
            "density_all": 0.0 if len(frame) < 2 else 2 * n_edges / (len(frame) * (len(frame) - 1)),
            "density_incident": 0.0 if n_incident < 2 else 2 * n_edges / (n_incident * (n_incident - 1)),
        })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, column, label in zip(axes, ["n_nodes_incident", "n_edges", "density_incident"], ["Nodes with >=1 edge", "Edges", "Density (incident nodes)"]):
        ax.plot(sweep["threshold"], sweep[column], linewidth=2)
        ax.axvline(float(task["threshold"]), color="crimson", linestyle="--", linewidth=1.5)
        ax.set(xlabel="Cosine-similarity threshold", ylabel=label)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "threshold_sweep.png", dpi=300)
    plt.close(fig)

    delta = float(cfg["hspn"]["adjacent_delta"])
    selected_threshold = float(task["threshold"])
    thresholds = sorted({round(max(0.0, selected_threshold - delta), 2), round(selected_threshold, 2), round(min(1.0, selected_threshold + delta), 2)})
    seeds = list(range(n_seeds))
    baseline_seed = int(cfg["base_seed"])
    if baseline_seed not in seeds:
        baseline_seed = seeds[0]
    all_results: dict[float, dict] = {}
    partition_tensors, candidate_tensors = [], []
    candidate_records, run_records, graph_records = [], [], []
    for threshold in thresholds:
        graph = graph_at_threshold(len(frame), i_idx, j_idx, similarities, threshold)
        components = list(nx.connected_components(graph))
        pagerank_map = nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=1000)
        pagerank = np.array([pagerank_map[i] for i in range(len(frame))])
        pd.DataFrame({"row_index": np.arange(len(frame)), "ID": ids, "pagerank": pagerank}).to_csv(out / f"pagerank_th{threshold:.2f}.csv", index=False)
        partitions, candidates, modularities, n_communities = {}, {}, {}, {}
        for seed in seeds:
            if graph.number_of_edges() == 0:
                communities = [{i} for i in range(len(frame))]
                modularity = float("nan")
            else:
                communities = list(nx.community.louvain_communities(graph, weight="weight", seed=seed))
                modularity = float(nx.community.modularity(graph, communities, weight="weight"))
            labels = partition_labels(communities, len(frame))
            chosen = historical_candidates(labels, pagerank, predicted)
            partitions[seed] = labels
            candidates[seed] = chosen
            modularities[seed] = modularity
            n_communities[seed] = len(communities)
            run_records.append({
                "key": task["key"], "threshold": threshold, "seed": seed,
                "modularity": modularity, "n_communities": len(communities),
                "n_candidates": len(chosen),
            })
        pair_ari, pair_candidate_jaccard = [], []
        for first, second in itertools.combinations(seeds, 2):
            pair_ari.append(float(adjusted_rand_score(partitions[first], partitions[second])))
            pair_candidate_jaccard.append(jaccard(candidates[first], candidates[second]))
        frequency = np.zeros(len(frame), dtype=float)
        for chosen in candidates.values():
            if chosen:
                frequency[list(chosen)] += 1
        frequency /= len(seeds)
        candidate_records.extend({
            "key": task["key"], "threshold": threshold, "row_index": i, "ID": ids[i],
            "selection_frequency": frequency[i], "predicted": predicted[i], "pagerank": pagerank[i],
        } for i in range(len(frame)))
        partition_tensors.append(np.stack([partitions[seed] for seed in seeds]).astype(np.int32))
        candidate_tensors.append(np.stack([
            np.isin(np.arange(len(frame)), list(candidates[seed])) for seed in seeds
        ]).astype(np.uint8))
        graph_records.append({
            "key": task["key"], "threshold": threshold, "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(), "n_components": len(components),
            "largest_component_fraction": max(map(len, components)) / len(frame),
            **summarize_pairwise(pair_ari, "partition_ari_pairwise"),
            **summarize_pairwise(pair_candidate_jaccard, "candidate_jaccard_pairwise"),
        })
        all_results[threshold] = {"partitions": partitions, "candidates": candidates, "pagerank": pagerank}

    selected = all_results[round(selected_threshold, 2)]
    cross_rows = []
    for threshold in thresholds:
        if np.isclose(threshold, selected_threshold):
            continue
        other = all_results[threshold]
        aris = [adjusted_rand_score(selected["partitions"][seed], other["partitions"][seed]) for seed in seeds]
        jac = [jaccard(selected["candidates"][seed], other["candidates"][seed]) for seed in seeds]
        ari_mean, ari_low, ari_high = percentile_ci(aris)
        jac_mean, jac_low, jac_high = percentile_ci(jac)
        cross_rows.append({
            "key": task["key"], "selected_threshold": selected_threshold, "comparison_threshold": threshold,
            "partition_ari_mean": ari_mean, "partition_ari_ci95_low": ari_low, "partition_ari_ci95_high": ari_high,
            "candidate_jaccard_mean": jac_mean, "candidate_jaccard_ci95_low": jac_low, "candidate_jaccard_ci95_high": jac_high,
            "pagerank_spearman": safe_spearman(selected["pagerank"], other["pagerank"]),
        })
    pd.DataFrame(run_records).to_csv(out / "louvain_runs.csv", index=False)
    pd.DataFrame(graph_records).to_csv(out / "threshold_stability_summary.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(out / "adjacent_threshold_comparison.csv", index=False)
    pd.DataFrame(candidate_records).to_csv(out / "candidate_selection_frequency.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"row_index": np.arange(len(frame)), "ID": ids}).to_csv(out / "row_order.csv", index=False)
    np.savez_compressed(
        out / "louvain_partitions_and_candidates.npz",
        thresholds=np.asarray(thresholds, dtype=float),
        seeds=np.asarray(seeds, dtype=int),
        partitions=np.stack(partition_tensors),
        candidates=np.stack(candidate_tensors),
    )
    write_completed_output(out, fingerprint, required, {
        "key": task["key"], "n": len(frame), "n_features": len(features),
        "thresholds_for_stability": thresholds, "n_louvain_seeds": n_seeds,
        "baseline_seed": baseline_seed,
        "candidate_rule": "lower half of modules by mean predicted, then upper half of nodes by PageRank",
    })
    print(f"completed HSPN stability: {task['key']}")


if __name__ == "__main__":
    main()
