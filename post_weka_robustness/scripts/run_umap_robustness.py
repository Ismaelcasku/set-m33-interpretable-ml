#!/usr/bin/env python3
"""Robustness analysis around the recovered grouped Euclidean UMAP."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from setm33robust.config import load_config, resolve_results_root, umap_jobs
from setm33robust.io import (
    analysis_fingerprint, begin_output, completed_output, sha256_file,
    write_completed_output,
)
from setm33robust.metrics import (
    balanced_subset_indices,
    bootstrap_mean_ci,
    effective_dimension,
    knn_indices,
    mean_knn_overlap,
)


def slug(job: dict) -> str:
    md = re.sub(r"[^0-9a-zA-Z]+", "p", f"{job['min_dist']:g}").strip("p")
    return f"{job['species']}__grouped_euclidean__nn{job['n_neighbors']}__md{md}"


def finite_silhouette(values: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
        return float("nan"), np.full(len(labels), np.nan)
    samples = silhouette_samples(values, labels, metric="euclidean")
    return float(np.mean(samples)), samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap-task-index", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated override for smoke tests")
    args = parser.parse_args()
    cfg = load_config(args.config)
    jobs = umap_jobs(cfg)
    if args.umap_task_index < 0 or args.umap_task_index >= len(jobs):
        raise IndexError(f"UMAP task index outside 0..{len(jobs)-1}")
    job = jobs[args.umap_task_index]
    section = cfg["umap"]
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [int(x) for x in section["seeds"]]
    primary_seed = int(section["primary_seed"])
    if primary_seed not in seeds:
        seeds = [primary_seed, *seeds]
    results = resolve_results_root(cfg)
    historical = results / "historical_umap" / job["species"]
    X_path = historical / "grouped_shap.npy"
    row_path = historical / "row_order.csv"
    if not X_path.exists() or not row_path.exists():
        raise FileNotFoundError(f"historical grouped space missing for {job['species']}")
    X = np.load(X_path).astype(float)
    metadata = pd.read_csv(row_path, dtype={"ID": str})
    if len(X) != len(metadata):
        raise ValueError("grouped SHAP matrix and row order differ")
    fingerprint = analysis_fingerprint({
        "stage": "historical_umap_robustness", "job": job, "seeds": seeds,
        "input_sha256": sha256_file(X_path),
        "config": {
            key: section[key] for key in [
                "metric", "knn_k", "metric_max_rows_per_charge",
                "metric_bootstrap_replicates", "hdbscan_min_cluster_size",
                "hdbscan_min_samples", "primary_seed",
            ]
        },
        "geometry_evaluation": "all_candidates",
        "charge_dependent_evaluation": "fixed_charge_balanced_subset",
        "knn_implementation": "first_k_nonself_neighbors__mean_intersection_over_k__v3",
    })
    out = results / "umap_robustness" / slug(job)
    required = ["metrics.csv", "charge_metrics.csv", "metric_subset.csv", "run_arrays.npz"]
    if completed_output(out, fingerprint, required):
        print(f"skip completed UMAP robustness: {slug(job)}")
        return
    out.mkdir(parents=True, exist_ok=True)
    begin_output(out, fingerprint, {"species": job["species"], **job})

    labels_charge = metadata["cationic_charge"].to_numpy(dtype=int)
    configured_max = section.get("metric_max_rows_per_charge")
    max_per_charge = len(labels_charge) if configured_max is None else int(configured_max)
    charge_metric_idx = balanced_subset_indices(labels_charge, max_per_charge, int(cfg["base_seed"]))
    X_charge_metric = X[charge_metric_idx]
    charge_metric = labels_charge[charge_metric_idx]
    pca = PCA(n_components=2, random_state=int(cfg["base_seed"])).fit_transform(X)

    # Geometry-fidelity metrics are evaluated on the complete candidate library.
    # The charge-balanced subset is reserved for charge-dependent summaries only.
    geometry_k = min(int(section["knn_k"]), len(X) - 1, (len(X) - 1) // 2)
    original_neighbors = knn_indices(X, geometry_k, metric="euclidean")
    grouped_sil, grouped_sil_samples = finite_silhouette(X_charge_metric, charge_metric)
    pca_sil, pca_sil_samples = finite_silhouette(pca[charge_metric_idx], charge_metric)
    reps = int(section["metric_bootstrap_replicates"])
    grouped_sil_stats = bootstrap_mean_ci(grouped_sil_samples, reps, int(cfg["base_seed"]))
    pca_sil_stats = bootstrap_mean_ci(pca_sil_samples, reps, int(cfg["base_seed"]) + 1)
    embeddings, clusterings, rows, charge_rows = [], [], [], []
    reference_embedding_neighbors = None
    reference_hdbscan = None
    for seed in seeds:
        embedding = umap.UMAP(
            n_components=2,
            n_neighbors=min(int(job["n_neighbors"]), len(X) - 1),
            min_dist=float(job["min_dist"]),
            metric="euclidean",
            random_state=seed,
            n_jobs=1,
        ).fit_transform(X).astype(np.float32)
        embedding_neighbors = knn_indices(embedding, geometry_k, metric="euclidean")
        metric_cluster_labels = np.full(len(charge_metric_idx), -1, dtype=np.int32)
        counts, noises, hdb_silhouettes, all_radii = [], [], [], []
        label_offset = 0
        for charge in np.unique(charge_metric):
            local_positions = np.flatnonzero(charge_metric == charge)
            points = embedding[charge_metric_idx[local_positions]]
            local_labels = hdbscan.HDBSCAN(
                min_cluster_size=int(section["hdbscan_min_cluster_size"]),
                min_samples=int(section["hdbscan_min_samples"]),
                metric="euclidean",
            ).fit_predict(points).astype(np.int32)
            clusters = np.unique(local_labels[local_labels >= 0])
            mapped = local_labels.copy()
            mapped[mapped >= 0] += label_offset
            if len(clusters):
                label_offset += int(local_labels.max()) + 1
            metric_cluster_labels[local_positions] = mapped
            non_noise = local_labels >= 0
            if len(clusters) >= 2 and non_noise.sum() > len(clusters):
                hdb_sil = float(silhouette_score(points[non_noise], local_labels[non_noise]))
            else:
                hdb_sil = float("nan")
            radii = []
            for cluster in clusters:
                cluster_points = points[local_labels == cluster]
                radii.extend(np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1).tolist())
            radius_mean, radius_low, radius_high = bootstrap_mean_ci(
                np.asarray(radii), reps, int(cfg["base_seed"]) + seed * 100 + int(charge)
            )
            counts.append(len(clusters))
            noises.append(float(np.mean(local_labels < 0)))
            hdb_silhouettes.append(hdb_sil)
            all_radii.extend(radii)
            charge_rows.append({
                **job, "seed": seed, "charge": int(charge), "n_sub": len(points),
                "hdbscan_n_clusters": len(clusters),
                "hdbscan_noise_fraction": float(np.mean(local_labels < 0)),
                "hdbscan_silhouette": hdb_sil,
                "intra_cluster_radius": radius_mean,
                "intra_cluster_radius_ci95_low": radius_low,
                "intra_cluster_radius_ci95_high": radius_high,
            })
        if seed == primary_seed:
            reference_embedding_neighbors = embedding_neighbors
            reference_hdbscan = metric_cluster_labels.copy()
        umap_sil, umap_sil_samples = finite_silhouette(embedding[charge_metric_idx], charge_metric)
        umap_sil_mean, umap_sil_low, umap_sil_high = bootstrap_mean_ci(
            umap_sil_samples, reps, int(cfg["base_seed"]) + seed + 100
        )
        radius_mean, radius_low, radius_high = bootstrap_mean_ci(
            np.asarray(all_radii), reps, int(cfg["base_seed"]) + seed + 10
        )
        finite_hdb = [value for value in hdb_silhouettes if np.isfinite(value)]
        rows.append({
            **job, "seed": seed, "n_rows": len(X), "n_groups": X.shape[1],
            "geometry_metric_n": len(X),
            "charge_balanced_subset_n": len(charge_metric_idx),
            "knn_k": geometry_k,
            "effective_dimension_grouped_shap": effective_dimension(X),
            "trustworthiness_grouped_shap": float(
                trustworthiness(X, embedding, n_neighbors=geometry_k, metric="euclidean")
            ),
            "knn_preservation_grouped_shap": mean_knn_overlap(original_neighbors, embedding_neighbors),
            "charge_silhouette_grouped_shap": grouped_sil,
            "charge_silhouette_grouped_shap_ci95_low": grouped_sil_stats[1],
            "charge_silhouette_grouped_shap_ci95_high": grouped_sil_stats[2],
            "charge_silhouette_pca": pca_sil,
            "charge_silhouette_pca_ci95_low": pca_sil_stats[1],
            "charge_silhouette_pca_ci95_high": pca_sil_stats[2],
            "charge_silhouette_umap": umap_sil_mean,
            "charge_silhouette_umap_ci95_low": umap_sil_low,
            "charge_silhouette_umap_ci95_high": umap_sil_high,
            "hdbscan_n_clusters_mean_across_charges": float(np.mean(counts)),
            "hdbscan_noise_fraction_mean_across_charges": float(np.mean(noises)),
            "hdbscan_silhouette_mean_across_charges": float(np.mean(finite_hdb)) if finite_hdb else float("nan"),
            "intra_cluster_radius": radius_mean,
            "intra_cluster_radius_ci95_low": radius_low,
            "intra_cluster_radius_ci95_high": radius_high,
        })
        embeddings.append(embedding)
        clusterings.append(metric_cluster_labels)
    if reference_embedding_neighbors is None or reference_hdbscan is None:
        raise RuntimeError("primary UMAP seed was not executed")
    for i, seed in enumerate(seeds):
        neighbors = knn_indices(embeddings[i], geometry_k, metric="euclidean")
        rows[i]["knn_overlap_vs_primary_seed"] = mean_knn_overlap(reference_embedding_neighbors, neighbors)
        charge_aris = []
        for charge in np.unique(charge_metric):
            local = charge_metric == charge
            ari = float(adjusted_rand_score(reference_hdbscan[local], clusterings[i][local]))
            charge_aris.append(ari)
            matching = next(row for row in charge_rows if row["seed"] == seed and row["charge"] == int(charge))
            matching["hdbscan_ari_vs_primary_seed"] = ari
        rows[i]["hdbscan_ari_vs_primary_seed_mean_across_charges"] = float(np.mean(charge_aris))
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(charge_rows).to_csv(out / "charge_metrics.csv", index=False)
    pd.DataFrame({
        "row_index": charge_metric_idx,
        "ID": metadata.iloc[charge_metric_idx]["ID"].to_numpy(),
        "cationic_charge": labels_charge[charge_metric_idx],
    }).to_csv(out / "metric_subset.csv", index=False)
    np.savez_compressed(
        out / "run_arrays.npz",
        embeddings=np.stack(embeddings),
        hdbscan_labels_metric_subset=np.stack(clusterings),
        seeds=np.asarray(seeds, dtype=int),
        pca=pca.astype(np.float32),
    )
    write_completed_output(out, fingerprint, required, {
        **job,
        "seeds": seeds, "n_rows": len(X), "n_groups": X.shape[1],
        "input_space": "historical per-peptide mean SHAP within KMeans descriptor groups",
        "metric": "euclidean", "pre_umap_scaling": "none, matching recovered code",
        "geometry_metrics": "trustworthiness and kNN preservation/overlap evaluated on all candidates",
        "charge_metrics": "charge silhouettes and HDBSCAN stability evaluated on a fixed charge-balanced subset",
        "knn_definition": "first k non-self neighbors; mean |intersection|/k",
        "biological_interpretation_boundary": (
            "projection geometry and algorithmic clusters are not treated as biological classes"
        ),
    })
    print(f"completed historical-pipeline UMAP robustness: {slug(job)}")


if __name__ == "__main__":
    main()
