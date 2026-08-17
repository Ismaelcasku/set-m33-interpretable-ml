#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from setm33robust.config import load_config, load_tasks, resolve_data_root, resolve_results_root, umap_jobs
from setm33robust.io import dataset_path, read_json, write_json
from setm33robust.metrics import (
    jaccard,
    knn_indices,
    mean_knn_overlap,
    pairwise_jaccards,
    percentile_ci,
    ranks_desc,
    safe_spearman,
)


def umap_slug(job: dict) -> str:
    md = re.sub(r"[^0-9a-zA-Z]+", "p", f"{job['min_dist']:g}").strip("p")
    return f"{job['species']}__grouped_euclidean__nn{job['n_neighbors']}__md{md}"


def require(paths: list[Path], allow_partial: bool) -> list[Path]:
    missing = [path for path in paths if not path.exists()]
    if missing and not allow_partial:
        preview = "\n".join(f"  - {path}" for path in missing[:30])
        raise FileNotFoundError(f"{len(missing)} required outputs are missing:\n{preview}")
    return [path for path in paths if path.exists()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_shap_source(
    key: str,
    source: str,
    runs: pd.DataFrame,
    replicate_col: str,
    reference: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, dict]:
    pivot = runs.pivot(index=replicate_col, columns="descriptor", values="mean_abs_shap")
    ref = reference.set_index("descriptor")["mean_abs_shap_stability_subset"].reindex(pivot.columns)
    if ref.isna().any() or pivot.isna().any().any():
        raise ValueError(f"{key}/{source}: incomplete descriptor matrix")
    ref_rank = ranks_desc(ref.to_numpy())
    reference_top = set(ref.nlargest(min(top_k, len(ref))).index)
    ranking_correlations, jaccard_reference, top_sets, rank_vectors = [], [], [], []
    for _, row in pivot.iterrows():
        run_rank = ranks_desc(row.to_numpy())
        rank_vectors.append(run_rank)
        ranking_correlations.append(safe_spearman(ref_rank, run_rank))
        top = set(row.nlargest(min(top_k, len(row))).index)
        top_sets.append(top)
        jaccard_reference.append(jaccard(reference_top, top))
    feature_rows = []
    for descriptor in pivot.columns:
        mean, low, high = percentile_ci(pivot[descriptor])
        feature_rows.append({
            "key": key, "source": source, "descriptor": descriptor,
            "reference_mean_abs_shap": float(ref[descriptor]),
            "mean_abs_shap": mean, "ci95_low": low, "ci95_high": high,
            "top10_frequency": float(np.mean([descriptor in x for x in top_sets])),
        })
    pair_j = pairwise_jaccards(top_sets)
    pair_spearman = np.asarray([
        safe_spearman(first, second)
        for first, second in itertools.combinations(rank_vectors, 2)
    ])
    rank_stats = percentile_ci(ranking_correlations)
    ref_j_stats = percentile_ci(jaccard_reference)
    pair_stats = percentile_ci(pair_j)
    pair_spearman_stats = percentile_ci(pair_spearman)
    summary = {
        "key": key, "source": source, "n_runs": len(pivot), "n_descriptors": len(pivot.columns),
        "ranking_spearman_mean": rank_stats[0], "ranking_spearman_ci95_low": rank_stats[1], "ranking_spearman_ci95_high": rank_stats[2],
        "ranking_spearman_pairwise_mean": pair_spearman_stats[0], "ranking_spearman_pairwise_ci95_low": pair_spearman_stats[1], "ranking_spearman_pairwise_ci95_high": pair_spearman_stats[2],
        "top10_jaccard_vs_reference_mean": ref_j_stats[0], "top10_jaccard_vs_reference_ci95_low": ref_j_stats[1], "top10_jaccard_vs_reference_ci95_high": ref_j_stats[2],
        "top10_jaccard_pairwise_mean": pair_stats[0], "top10_jaccard_pairwise_ci95_low": pair_stats[1], "top10_jaccard_pairwise_ci95_high": pair_stats[2],
    }
    return pd.DataFrame(feature_rows), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks = load_tasks()
    results = resolve_results_root(cfg)
    summary_root = results / "summary"
    tables = summary_root / "tables"
    figures = summary_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    reference_rows = []
    reference_paths = require([results / "surrogate_reference" / key / "metrics.json" for key in tasks["key"]], args.allow_partial)
    for path in reference_paths:
        metrics = read_json(path)
        reference_rows.append({"key": path.parent.name, **metrics})
    pd.DataFrame(reference_rows).to_csv(tables / "surrogate_in_sample_fidelity.csv", index=False)

    partition_paths = require([results / "surrogate_cv" / key / "partition_variability_summary.csv" for key in tasks["key"]], args.allow_partial)
    partition = pd.concat([pd.read_csv(path).assign(key=path.parent.name) for path in partition_paths], ignore_index=True) if partition_paths else pd.DataFrame()
    candidate_paths = require([results / "surrogate_cv" / key / "candidate_bootstrap_summary.csv" for key in tasks["key"]], args.allow_partial)
    candidate = pd.concat([pd.read_csv(path).assign(key=path.parent.name) for path in candidate_paths], ignore_index=True) if candidate_paths else pd.DataFrame()
    if not partition.empty:
        partition.to_csv(tables / "surrogate_partition_variability.csv", index=False)
    if not candidate.empty:
        candidate.to_csv(tables / "surrogate_candidate_bootstrap_fidelity.csv", index=False)
        order = ["r2", "mae", "rmse", "spearman"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric in zip(axes.flat, order):
            part = candidate.loc[candidate["metric"].eq(metric)].set_index("key").reindex(tasks["key"])
            ax.errorbar(np.arange(len(part)), part["point_estimate"], yerr=[part["point_estimate"] - part["ci95_low"], part["ci95_high"] - part["point_estimate"]], fmt="o", capsize=3)
            ax.set_xticks(np.arange(len(part)), part.index, rotation=70, ha="right", fontsize=7)
            ax.set_title(metric.upper())
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "surrogate_cv_fidelity.png", dpi=300)
        plt.close(fig)

    top_k = int(cfg["shap"]["top_k"])
    n_seed_runs = len(cfg["shap"]["rf_seeds"])
    n_boot = int(cfg["shap"]["bootstrap_replicates"])
    chunk = int(cfg["shap"]["bootstrap_chunk_size"])
    n_chunks = (n_boot + chunk - 1) // chunk
    shap_feature_tables, shap_summaries = [], []
    for _, task in tasks.iterrows():
        key = str(task["key"])
        ref_path = results / "shap_reference" / key / "reference_importance.csv"
        if not ref_path.exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(ref_path)
        reference = pd.read_csv(ref_path)
        seed_paths = require([results / "shap_seeds" / key / f"seed_{i:03d}.csv" for i in range(n_seed_runs)], args.allow_partial)
        if seed_paths:
            seed_runs = pd.concat([pd.read_csv(path) for path in seed_paths], ignore_index=True)
            features, summary = aggregate_shap_source(key, "rf_seed", seed_runs, "seed_index", reference, top_k)
            shap_feature_tables.append(features)
            shap_summaries.append(summary)
        boot_paths = require([results / "shap_bootstrap" / key / f"chunk_{i:03d}_importance.csv.gz" for i in range(n_chunks)], args.allow_partial)
        if boot_paths:
            boot_runs = pd.concat([pd.read_csv(path) for path in boot_paths], ignore_index=True)
            features, summary = aggregate_shap_source(key, "bootstrap", boot_runs, "replicate", reference, top_k)
            shap_feature_tables.append(features)
            shap_summaries.append(summary)
            top = features.nlargest(min(15, len(features)), "reference_mean_abs_shap").sort_values("reference_mean_abs_shap")
            fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.32)))
            ax.errorbar(top["mean_abs_shap"], np.arange(len(top)), xerr=[top["mean_abs_shap"] - top["ci95_low"], top["ci95_high"] - top["mean_abs_shap"]], fmt="o", capsize=2)
            ax.set_yticks(np.arange(len(top)), top["descriptor"], fontsize=7)
            ax.set(xlabel="Bootstrap mean(|SHAP|), 95% percentile CI", title=key)
            fig.tight_layout()
            fig.savefig(figures / f"shap_bootstrap_{key}.png", dpi=300)
            plt.close(fig)
    if shap_feature_tables:
        pd.concat(shap_feature_tables, ignore_index=True).to_csv(tables / "shap_feature_stability.csv", index=False)
        pd.DataFrame(shap_summaries).to_csv(tables / "shap_ranking_stability.csv", index=False)

    hspn_tables = {
        "hspn_threshold_sweep.csv": "threshold_sweep.csv",
        "hspn_louvain_runs.csv": "louvain_runs.csv",
        "hspn_threshold_stability.csv": "threshold_stability_summary.csv",
        "hspn_adjacent_threshold_comparison.csv": "adjacent_threshold_comparison.csv",
    }
    for output_name, input_name in hspn_tables.items():
        paths = require([results / "hspn" / key / input_name for key in tasks["key"]], args.allow_partial)
        if paths:
            pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(tables / output_name, index=False)

    species = tasks["species"].drop_duplicates().tolist()
    historical_paths = require(
        [results / "historical_umap" / item / "UMAP_summary.csv" for item in species],
        args.allow_partial,
    )
    if historical_paths:
        pd.concat([pd.read_csv(path) for path in historical_paths], ignore_index=True).to_csv(
            tables / "historical_UMAP_summary.csv", index=False
        )
    jobs = umap_jobs(cfg)
    metric_paths = require([results / "umap_robustness" / umap_slug(job) / "metrics.csv" for job in jobs], args.allow_partial)
    umap_metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True) if metric_paths else pd.DataFrame()
    if not umap_metrics.empty:
        global_rows = []
        for job in jobs:
            directory = results / "umap_robustness" / umap_slug(job)
            if not (directory / "run_arrays.npz").exists():
                continue
            reference_job = {
                "species": job["species"],
                "n_neighbors": int(cfg["umap"]["primary_n_neighbors"]),
                "min_dist": float(cfg["umap"]["primary_min_dist"]),
            }
            ref_dir = results / "umap_robustness" / umap_slug(reference_job)
            if not (ref_dir / "run_arrays.npz").exists():
                continue
            current = np.load(directory / "run_arrays.npz")
            reference = np.load(ref_dir / "run_arrays.npz")
            current_rows = pd.read_csv(results / "historical_umap" / job["species"] / "row_order.csv", dtype={"ID": str})
            subset = pd.read_csv(ref_dir / "metric_subset.csv")["row_index"].to_numpy(dtype=int)
            ref_seeds = reference["seeds"].tolist()
            ref_seed_idx = ref_seeds.index(int(cfg["umap"]["primary_seed"]))
            ref_emb = reference["embeddings"][ref_seed_idx]
            ref_labels = reference["hdbscan_labels_metric_subset"][ref_seed_idx]
            subset_charges = current_rows.iloc[subset]["cationic_charge"].to_numpy()
            geometry_k = min(
                int(cfg["umap"]["knn_k"]),
                len(current_rows) - 1,
                (len(current_rows) - 1) // 2,
            )
            ref_neighbors = knn_indices(ref_emb, geometry_k, metric="euclidean")
            for index, seed in enumerate(current["seeds"].tolist()):
                charge_aris = [
                    adjusted_rand_score(
                        ref_labels[subset_charges == charge],
                        current["hdbscan_labels_metric_subset"][index][subset_charges == charge],
                    )
                    for charge in np.unique(subset_charges)
                ]
                global_rows.append({
                    **job, "seed": int(seed),
                    "knn_overlap_vs_primary_configuration": mean_knn_overlap(
                        ref_neighbors,
                        knn_indices(current["embeddings"][index], geometry_k, metric="euclidean"),
                    ),
                    "hdbscan_ari_vs_primary_configuration": float(np.mean(charge_aris)),
                })
        comparisons = pd.DataFrame(global_rows)
        if not comparisons.empty:
            umap_metrics = umap_metrics.merge(comparisons, on=["umap_task_index", "species", "n_neighbors", "min_dist", "seed"], how="left", validate="one_to_one")
        umap_metrics.to_csv(tables / "umap_run_metrics.csv", index=False)
        metric_columns = [
            "trustworthiness_grouped_shap", "knn_preservation_grouped_shap", "knn_overlap_vs_primary_seed",
            "knn_overlap_vs_primary_configuration", "charge_silhouette_grouped_shap",
            "charge_silhouette_pca", "charge_silhouette_umap", "hdbscan_n_clusters_mean_across_charges",
            "hdbscan_noise_fraction_mean_across_charges", "hdbscan_ari_vs_primary_seed_mean_across_charges",
            "hdbscan_ari_vs_primary_configuration", "intra_cluster_radius",
        ]
        grouped_rows = []
        group_cols = ["species", "n_neighbors", "min_dist"]
        for group, part in umap_metrics.groupby(group_cols):
            row = dict(zip(group_cols, group))
            row["n_seeds"] = len(part)
            for metric in metric_columns:
                if metric in part:
                    mean, low, high = percentile_ci(part[metric])
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_ci95_low"] = low
                    row[f"{metric}_ci95_high"] = high
            grouped_rows.append(row)
        pd.DataFrame(grouped_rows).to_csv(tables / "umap_configuration_summary.csv", index=False)
        charge_paths = [results / "umap_robustness" / umap_slug(job) / "charge_metrics.csv" for job in jobs]
        charge_paths = [path for path in charge_paths if path.exists()]
        if charge_paths:
            pd.concat([pd.read_csv(path) for path in charge_paths], ignore_index=True).to_csv(tables / "umap_hdbscan_metrics_by_charge.csv", index=False)

    data_root = resolve_data_root(cfg)
    inputs = []
    for _, task in tasks.iterrows():
        path = dataset_path(data_root, task)
        inputs.append({"key": task["key"], "path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "networkx", "shap", "umap-learn", "hdbscan", "numba", "llvmlite"]:
        versions[name] = importlib.metadata.version(name)
    write_json(summary_root / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": "2.0.1", "config": cfg,
        "tasks_sha256": file_sha256(ROOT / "config" / "tasks.csv"),
        "inputs": inputs, "python": platform.python_version(), "packages": versions,
        "partial_aggregation": bool(args.allow_partial),
    })
    methods = """# Methods text for the robustness audit

The archived candidate tables containing the predictions of the selected WEKA models were used as the sole input to the post-WEKA interpretability audit. For each organism/charge library, a scikit-learn Random Forest regressor (100 trees, maximum depth 10, random seed 42) was trained to approximate the archived WEKA `predicted` values from the selected-pool descriptors. Surrogate fidelity was quantified in the full fit and by ten repetitions of shuffled five-fold cross-validation using R2, MAE, RMSE and Spearman correlation. The model seed was fixed across folds. The distribution across repeated partitions was reported descriptively because the repetitions reuse the same candidates. Inferential 95% intervals were instead obtained by paired bootstrap resampling of candidates from their archived WEKA values and mean out-of-fold surrogate predictions.

TreeSHAP values explain this Python surrogate and not the serialized WEKA trees. TreeExplainer used `feature_perturbation="tree_path_dependent"`. Model-seed stability was assessed using 20 Random Forest seeds. Sampling stability was assessed independently using 200 nonparametric bootstrap training samples while fixing the Random Forest seed at 42. Every stability model explained the same deterministic response-stratified subset of at most 1,000 candidates. Descriptor-level mean absolute SHAP values, empirical 95% intervals, correlations with the reference ranking, pairwise Spearman rank correlations, top-10 selection frequencies and top-10 Jaccard overlaps were calculated.

For HSPN sensitivity, descriptors were standardized within each candidate library using the population standard deviation and pairwise cosine similarity was calculated. The full 0.05-0.95 threshold sweep was retained. At the selected threshold and at +/-0.05, weighted Louvain community detection was repeated for 100 seeds. Modularity, adjusted Rand index, PageRank rank correlation and stability of the recovered candidate-selection rule were summarized. These Python repetitions constitute a sensitivity analysis and do not replace the original Gephi-derived labels.

The dimensionality analysis reproduced the recovered historical workflow. Charge-specific SHAP tables were concatenated on the union of descriptors. Mean signed SHAP values were calculated for every descriptor and cationic charge, absent descriptor/charge combinations were represented by zero, and these descriptor-by-charge dynamics were standardized. Descriptors were grouped by KMeans using six groups for E. coli, six for P. aeruginosa and three for K. pneumoniae (random seed 42; ten initializations). Each candidate was represented by its mean SHAP value within each descriptor group, using the descriptors available in its charge-specific pool; completely absent groups were represented by zero. This grouped matrix was projected without additional scaling by Euclidean UMAP (15 neighbors, minimum distance 0.1, two components, random seed 42) without subsampling.

Robustness was assessed on this same grouped SHAP representation, rather than on a separate direct-SHAP/cosine pipeline. Euclidean UMAP was repeated across neighbor settings of 5, 15 and 50, minimum distances of 0.0, 0.1 and 0.5, and ten random seeds. Trustworthiness, nearest-neighbor preservation and between-run neighborhood overlap were evaluated on the complete candidate library. Neighborhood preservation and overlap were defined as the mean fraction of the k=15 first non-self neighbors shared between two spaces or runs, |N1(i) intersection N2(i)|/k. Charge-dependent metrics were evaluated separately on a fixed charge-balanced subset constructed using the size of the smallest charge library within each organism. Charge-label silhouettes were calculated in grouped SHAP, two-component PCA and UMAP spaces, and HDBSCAN cluster count, noise fraction, silhouette, adjusted-Rand stability and mean within-cluster radius with bootstrap 95% intervals were recorded within this balanced subset. UMAP geometry and algorithmic clusters were not interpreted as direct evidence of biological classes.
"""
    (summary_root / "manuscript_methods.md").write_text(methods, encoding="utf-8")
    results_text = """# Results reporting template

Use the generated tables to report ranges and task-specific estimates. Do not fill missing values by hand.

- Surrogate fidelity: `tables/surrogate_in_sample_fidelity.csv`, `tables/surrogate_partition_variability.csv` and `tables/surrogate_candidate_bootstrap_fidelity.csv`.
- SHAP stability: `tables/shap_ranking_stability.csv` and `tables/shap_feature_stability.csv`.
- Network sensitivity: `tables/hspn_threshold_stability.csv` and `tables/hspn_adjacent_threshold_comparison.csv`.
- Historical dimensionality analysis: `tables/historical_UMAP_summary.csv`.
- Dimensionality robustness on that same grouped Euclidean pipeline: `tables/umap_run_metrics.csv` and `tables/umap_configuration_summary.csv`.

Recommended interpretation: describe whether conclusions are stable under resampling and parameter perturbation. Avoid assigning biological meaning to UMAP distances, HDBSCAN labels or Louvain modules without independent validation.
"""
    (summary_root / "manuscript_results_template.md").write_text(results_text, encoding="utf-8")
    write_json(summary_root / "aggregation.json", {"status": "completed", "partial": bool(args.allow_partial)})
    print(f"aggregation completed: {summary_root}")


if __name__ == "__main__":
    main()
