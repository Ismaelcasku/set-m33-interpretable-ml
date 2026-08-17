#!/usr/bin/env python3
"""Reconstruct the descriptor-grouped Euclidean UMAP recovered from old code."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from setm33robust.config import load_config, load_tasks, resolve_results_root
from setm33robust.io import (
    analysis_fingerprint,
    begin_output,
    completed_output,
    load_shap_reference,
    sha256_file,
    write_completed_output,
)


def species_at(index: int) -> str:
    species = load_tasks()["species"].drop_duplicates().tolist()
    if index < 0 or index >= len(species):
        raise IndexError(f"species index {index} outside 0..{len(species)-1}")
    return species[index]


def build_historical_space(results: Path, species: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return grouped peptide SHAP, descriptor dynamics and descriptor groups.

    This deliberately mirrors the recovered pandas workflow: charge-specific
    SHAP frames are concatenated on the union of descriptor columns, descriptor
    dynamics are zero-filled only for KMeans, and per-peptide group means use
    pandas skip-NaN semantics before a final zero fill.
    """
    selected = load_tasks().loc[lambda x: x["species"].eq(species)].sort_values("charge")
    frames = []
    input_hashes = []
    for _, task in selected.iterrows():
        directory = results / "shap_reference" / str(task["key"])
        values, features, rows = load_shap_reference(directory)
        shap_frame = pd.DataFrame(np.asarray(values), columns=features)
        shap_frame["ID"] = rows["ID"].astype(str).to_numpy()
        shap_frame["predicted"] = rows["predicted"].to_numpy(dtype=float)
        shap_frame["sequence"] = rows["sequence"].astype(str).to_numpy()
        shap_frame["cationic_charge"] = int(task["charge"])
        shap_frame["key"] = str(task["key"])
        frames.append(shap_frame)
        input_hashes.append(sha256_file(directory / "shap_values.npy"))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    metadata = {"ID", "predicted", "sequence", "cationic_charge", "key"}
    descriptors = sorted(set(combined.columns) - metadata)
    dynamics = combined.groupby("cationic_charge", sort=True)[descriptors].mean().transpose()
    dynamics_filled = dynamics.fillna(0.0)
    scaled_dynamics = StandardScaler().fit_transform(dynamics_filled.to_numpy(dtype=float))
    k = int(cfg["umap"]["descriptor_group_k"][species])
    kmeans = KMeans(
        n_clusters=k,
        random_state=int(cfg["umap"]["kmeans_seed"]),
        n_init=int(cfg["umap"]["kmeans_n_init"]),
    )
    labels = kmeans.fit_predict(scaled_dynamics)
    groups = pd.DataFrame({"descriptor": descriptors, "cluster": labels.astype(int)})
    grouped = pd.DataFrame(index=combined.index)
    for cluster in sorted(groups["cluster"].unique()):
        members = groups.loc[groups["cluster"].eq(cluster), "descriptor"].tolist()
        grouped[f"group_{cluster}"] = combined[members].mean(axis=1, skipna=True)
    grouped = grouped.fillna(0.0)
    for column in ["cationic_charge", "predicted", "ID", "sequence", "key"]:
        grouped[column] = combined[column].to_numpy()
    groups["kmeans_inertia"] = float(kmeans.inertia_)
    groups.attrs["descriptor_silhouette"] = (
        float(silhouette_score(scaled_dynamics, labels)) if len(np.unique(labels)) > 1 else float("nan")
    )
    groups.attrs["input_hashes"] = input_hashes
    return grouped, dynamics, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-index", type=int, required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    species = species_at(args.species_index)
    results = resolve_results_root(cfg)
    selected = load_tasks().loc[lambda x: x["species"].eq(species)].sort_values("charge")
    input_hashes = {
        str(task["key"]): sha256_file(results / "shap_reference" / str(task["key"]) / "shap_values.npy")
        for _, task in selected.iterrows()
    }
    historical_cfg = {
        key: cfg["umap"][key]
        for key in [
            "descriptor_group_k", "kmeans_seed", "kmeans_n_init", "metric",
            "primary_n_neighbors", "primary_min_dist", "primary_seed",
        ]
    }
    fingerprint = analysis_fingerprint({
        "stage": "historical_umap", "species": species,
        "inputs": input_hashes, "config": historical_cfg,
    })
    out = results / "historical_umap" / species
    required = [
        "grouped_shap.csv.gz", "grouped_shap.npy", "row_order.csv",
        "descriptor_charge_dynamics_with_nan.csv",
        "descriptor_charge_dynamics_zero_filled.csv", "descriptor_groups.csv",
        "historical_embedding.csv", "historical_embedding.npy", "UMAP_summary.csv",
    ]
    if completed_output(out, fingerprint, required):
        print(f"skip completed historical UMAP: {species}")
        return
    out.mkdir(parents=True, exist_ok=True)
    begin_output(out, fingerprint, {"species": species})
    grouped, dynamics, groups = build_historical_space(results, species, cfg)
    group_columns = [column for column in grouped.columns if column.startswith("group_")]
    X = grouped[group_columns].to_numpy(dtype=float)
    reducer = umap.UMAP(
        n_neighbors=min(int(cfg["umap"]["primary_n_neighbors"]), len(X) - 1),
        min_dist=float(cfg["umap"]["primary_min_dist"]),
        n_components=2,
        metric="euclidean",
        random_state=int(cfg["umap"]["primary_seed"]),
        n_jobs=1,
    )
    embedding = reducer.fit_transform(X).astype(np.float32)
    dynamics.to_csv(out / "descriptor_charge_dynamics_with_nan.csv")
    dynamics.fillna(0.0).to_csv(out / "descriptor_charge_dynamics_zero_filled.csv")
    groups.to_csv(out / "descriptor_groups.csv", index=False)
    grouped.to_csv(out / "grouped_shap.csv.gz", index=False, compression="gzip")
    np.save(out / "grouped_shap.npy", X.astype(np.float32))
    row_order = grouped[["ID", "key", "cationic_charge", "predicted", "sequence"]].copy()
    row_order.to_csv(out / "row_order.csv", index=False)
    embedding_table = row_order.copy()
    embedding_table["UMAP1"] = embedding[:, 0]
    embedding_table["UMAP2"] = embedding[:, 1]
    embedding_table.to_csv(out / "historical_embedding.csv", index=False)
    np.save(out / "historical_embedding.npy", embedding)
    summary = pd.DataFrame([{
        "species": species,
        "n_analogs": len(grouped),
        "n_union_descriptors": len(groups),
        "descriptor_groups_k": len(group_columns),
        "kmeans_seed": int(cfg["umap"]["kmeans_seed"]),
        "kmeans_n_init": int(cfg["umap"]["kmeans_n_init"]),
        "descriptor_silhouette": groups.attrs["descriptor_silhouette"],
        "umap_n_neighbors": int(cfg["umap"]["primary_n_neighbors"]),
        "umap_min_dist": float(cfg["umap"]["primary_min_dist"]),
        "umap_metric": "euclidean",
        "umap_seed": int(cfg["umap"]["primary_seed"]),
        "subsampling": "none",
    }])
    summary.to_csv(out / "UMAP_summary.csv", index=False)
    write_completed_output(out, fingerprint, required, {
        "species": species,
        "n_rows": len(grouped), "n_union_descriptors": len(groups),
        "descriptor_groups_k": len(group_columns), "input_hashes": input_hashes,
        "historical_definition": (
            "charge-mean signed SHAP descriptor dynamics; zero fill; StandardScaler; "
            "KMeans; per-peptide within-group mean; unscaled Euclidean UMAP"
        ),
    })
    print(f"completed historical grouped Euclidean UMAP: {species}")


if __name__ == "__main__":
    main()
