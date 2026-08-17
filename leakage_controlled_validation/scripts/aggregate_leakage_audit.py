#!/usr/bin/env python3
"""Validate and aggregate the paired leakage audit into analysis-ready tables."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_to_csv, atomic_write_json, load_config, project_root_from_file
from leakage_common import (
    audit_fold_dir,
    audit_fold_is_complete,
    audit_root,
    audit_task_from_id,
    extended_regression_metrics,
    load_audit_config,
    load_audit_manifest,
    total_audit_tasks,
)


METRICS = [
    "pearson_r",
    "spearman_rho",
    "mae",
    "rmse",
    "r2",
    "rae_percent",
    "rrse_percent",
    "within_1_twofold_percent",
    "within_2_twofold_percent",
    "positive_prediction_percent",
    "median_relative_error_percent",
]
HIGHER_IS_BETTER = {
    "pearson_r",
    "spearman_rho",
    "r2",
    "within_1_twofold_percent",
    "within_2_twofold_percent",
    "positive_prediction_percent",
}
COMPARISONS = [
    ("leakage_effect", "legacy_global", "nested_matched"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=None,
        help="Override config only for a diagnostic run; production default is 5000.",
    )
    return parser.parse_args()


def subset_frame(frame: pd.DataFrame, subset: str, cutoff: float) -> pd.DataFrame:
    if subset == "all_positive_mic":
        return frame
    if subset == "experimental_only":
        return frame.loc[frame["actual_mic_original"] < cutoff]
    raise ValueError(f"Unknown subset: {subset}")


def metrics_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    if len(frame) < 2:
        raise ValueError("At least two observations are required for metrics")
    return extended_regression_metrics(
        frame["actual_model_scale"], frame["predicted_model_scale"]
    )


def signed_advantage(metric: str, value_a: float, value_b: float) -> float:
    raw = value_a - value_b
    return raw if metric in HIGHER_IS_BETTER else -raw


def bootstrap_single_branch(
    frame: pd.DataFrame, reps: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    actual = frame["actual_model_scale"].to_numpy(dtype=float)
    predicted = frame["predicted_model_scale"].to_numpy(dtype=float)
    estimate = extended_regression_metrics(actual, predicted)
    draws = {metric: [] for metric in METRICS}
    for _ in range(reps):
        sampled = rng.integers(0, len(frame), size=len(frame))
        values = extended_regression_metrics(actual[sampled], predicted[sampled])
        for metric in METRICS:
            draws[metric].append(values[metric])
    rows = []
    for metric in METRICS:
        values = np.asarray(draws[metric], dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "metric": metric,
                "estimate_on_mean_oof": estimate[metric],
                "ci95_lower": float(np.percentile(values, 2.5)),
                "ci95_upper": float(np.percentile(values, 97.5)),
                "bootstrap_reps_valid": int(len(values)),
                "bootstrap_unit": "positive_cohort_row",
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["dataset", "row_index", "source_pandas_row", "source_csv_line", "sample_id"]
    left = frame_a[keys + ["actual_model_scale", "predicted_model_scale"]].rename(
        columns={
            "actual_model_scale": "actual_a",
            "predicted_model_scale": "predicted_a",
        }
    )
    right = frame_b[keys + ["actual_model_scale", "predicted_model_scale"]].rename(
        columns={
            "actual_model_scale": "actual_b",
            "predicted_model_scale": "predicted_b",
        }
    )
    aligned = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(aligned) != len(frame_a) or len(aligned) != len(frame_b):
        raise RuntimeError("Paired branch predictions do not have identical row coverage")
    if not np.allclose(aligned["actual_a"], aligned["actual_b"], rtol=0, atol=1e-12):
        raise RuntimeError("Paired branches disagree on actual MIC values")
    actual = aligned["actual_a"].to_numpy(dtype=float)
    pred_a = aligned["predicted_a"].to_numpy(dtype=float)
    pred_b = aligned["predicted_b"].to_numpy(dtype=float)
    estimate_a = extended_regression_metrics(actual, pred_a)
    estimate_b = extended_regression_metrics(actual, pred_b)
    draws = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        sampled = rng.integers(0, len(aligned), size=len(aligned))
        values_a = extended_regression_metrics(actual[sampled], pred_a[sampled])
        values_b = extended_regression_metrics(actual[sampled], pred_b[sampled])
        for metric in METRICS:
            draws[metric].append(
                signed_advantage(metric, values_a[metric], values_b[metric])
            )
    rows = []
    for metric in METRICS:
        values = np.asarray(draws[metric], dtype=float)
        values = values[np.isfinite(values)]
        raw_difference = estimate_a[metric] - estimate_b[metric]
        rows.append(
            {
                "metric": metric,
                "estimate_branch_a": estimate_a[metric],
                "estimate_branch_b": estimate_b[metric],
                "raw_difference_a_minus_b": raw_difference,
                "signed_advantage_branch_a": signed_advantage(
                    metric, estimate_a[metric], estimate_b[metric]
                ),
                "signed_advantage_ci95_lower": float(np.percentile(values, 2.5)),
                "signed_advantage_ci95_upper": float(np.percentile(values, 97.5)),
                "bootstrap_reps_valid": int(len(values)),
                "bootstrap_unit": "paired_positive_cohort_row",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    project_config = load_config(root)
    audit_config = load_audit_config(root)
    manifest = load_audit_manifest(root, audit_config)
    signature = str(manifest["run_signature"])
    summary_dir = audit_root(root, audit_config) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    total = total_audit_tasks(project_config, audit_config)

    missing = []
    prediction_frames = []
    selected_frames = []
    fold_metric_rows = []
    for task_id in range(total):
        spec = audit_task_from_id(project_config, audit_config, task_id)
        if not audit_fold_is_complete(root, audit_config, spec, signature):
            missing.append(task_id)
            continue
        directory = audit_fold_dir(root, audit_config, spec)
        predictions = pd.read_csv(directory / "oof_predictions.csv")
        selected = pd.read_csv(directory / "selected_features.csv")
        prediction_frames.append(predictions)
        selected_frames.append(selected)
        fold_metric_rows.append(
            {
                "branch": spec.branch,
                "dataset": spec.base.dataset_name,
                "seed": spec.base.seed,
                "fold": spec.base.fold,
                **metrics_from_frame(predictions),
            }
        )
    if missing:
        preview = ",".join(map(str, missing[:40])) + ("..." if len(missing) > 40 else "")
        raise RuntimeError(
            f"Cannot aggregate: {len(missing)} tasks are incomplete ({preview}). "
            "Run scripts/check_leakage_progress.py."
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_selected = pd.concat(selected_frames, ignore_index=True)
    atomic_to_csv(all_predictions, summary_dir / "all_oof_predictions.csv")
    atomic_to_csv(all_selected, summary_dir / "all_selected_features.csv")
    atomic_to_csv(pd.DataFrame(fold_metric_rows), summary_dir / "metrics_by_fold.csv")

    expected_rows = {
        str(row["dataset"]): int(row["n_rows_positive_model_cohort"])
        for row in manifest["cohorts"]
    }
    cutoff = float(audit_config["operational_negative_mic"])
    subsets = ["all_positive_mic", "experimental_only"]
    seed_rows = []
    for branch in audit_config["branches"]:
        for dataset in [str(item["name"]) for item in project_config["datasets"]]:
            for seed in [int(item) for item in project_config["seeds"]]:
                frame = all_predictions.query(
                    "branch == @branch and dataset == @dataset and seed == @seed"
                ).sort_values("row_index")
                expected = expected_rows[dataset]
                if (
                    len(frame) != expected
                    or frame["row_index"].nunique() != expected
                    or frame["row_index"].astype(int).tolist() != list(range(expected))
                ):
                    raise RuntimeError(
                        f"Invalid OOF coverage for {branch}, {dataset}, seed {seed}: "
                        f"{len(frame)} rows/{frame['row_index'].nunique()} unique; expected {expected}"
                    )
                destination = (
                    summary_dir
                    / "per_seed"
                    / branch
                    / dataset
                    / f"seed_{seed:02d}_all_oof_predictions.csv"
                )
                atomic_to_csv(frame, destination)
                for subset in subsets:
                    analysis = subset_frame(frame, subset, cutoff)
                    seed_rows.append(
                        {
                            "branch": branch,
                            "dataset": dataset,
                            "seed": seed,
                            "analysis_subset": subset,
                            **metrics_from_frame(analysis),
                        }
                    )
    metrics_by_seed = pd.DataFrame(seed_rows)
    atomic_to_csv(metrics_by_seed, summary_dir / "metrics_by_seed.csv")

    summary_rows = []
    for keys, group in metrics_by_seed.groupby(
        ["branch", "dataset", "analysis_subset"], sort=False
    ):
        branch, dataset, subset = keys
        for metric in METRICS:
            summary_rows.append(
                {
                    "branch": branch,
                    "dataset": dataset,
                    "analysis_subset": subset,
                    "metric": metric,
                    "mean_across_seeds": float(group[metric].mean()),
                    "sd_across_seeds": float(group[metric].std(ddof=1)),
                    "minimum_across_seeds": float(group[metric].min()),
                    "maximum_across_seeds": float(group[metric].max()),
                    "n_seeds": int(group["seed"].nunique()),
                }
            )
    metric_summary = pd.DataFrame(summary_rows)
    atomic_to_csv(metric_summary, summary_dir / "metrics_summary_across_seeds.csv")

    grouping = [
        "branch",
        "dataset",
        "row_index",
        "source_pandas_row",
        "source_csv_line",
        "sample_id",
        "actual_model_scale",
        "actual_mic_original",
    ]
    mean_oof = (
        all_predictions.groupby(grouping, as_index=False, dropna=False)
        .agg(
            predicted_model_scale=("predicted_model_scale", "mean"),
            prediction_sd_across_seeds=("predicted_model_scale", "std"),
            prediction_min_across_seeds=("predicted_model_scale", "min"),
            prediction_max_across_seeds=("predicted_model_scale", "max"),
            n_seed_predictions=("predicted_model_scale", "size"),
        )
    )
    if not (mean_oof["n_seed_predictions"] == len(project_config["seeds"])).all():
        raise RuntimeError("Per-sample prediction uncertainty is missing one or more seeds")
    mean_oof["predicted_mic_original"] = np.power(
        10.0, mean_oof["predicted_model_scale"]
    )
    atomic_to_csv(mean_oof, summary_dir / "mean_oof_prediction_per_sample.csv")

    paired_seed_rows = []
    for comparison, branch_a, branch_b in COMPARISONS:
        for dataset in [str(item["name"]) for item in project_config["datasets"]]:
            for seed in [int(item) for item in project_config["seeds"]]:
                for subset in subsets:
                    a = metrics_by_seed.query(
                        "branch == @branch_a and dataset == @dataset and seed == @seed and analysis_subset == @subset"
                    ).iloc[0]
                    b = metrics_by_seed.query(
                        "branch == @branch_b and dataset == @dataset and seed == @seed and analysis_subset == @subset"
                    ).iloc[0]
                    for metric in METRICS:
                        paired_seed_rows.append(
                            {
                                "comparison": comparison,
                                "branch_a": branch_a,
                                "branch_b": branch_b,
                                "dataset": dataset,
                                "seed": seed,
                                "analysis_subset": subset,
                                "metric": metric,
                                "value_branch_a": float(a[metric]),
                                "value_branch_b": float(b[metric]),
                                "raw_difference_a_minus_b": float(a[metric] - b[metric]),
                                "signed_advantage_branch_a": signed_advantage(
                                    metric, float(a[metric]), float(b[metric])
                                ),
                            }
                        )
    paired_seed = pd.DataFrame(paired_seed_rows)
    atomic_to_csv(paired_seed, summary_dir / "paired_metric_differences_by_seed.csv")
    paired_seed_summary = (
        paired_seed.groupby(
            [
                "comparison",
                "branch_a",
                "branch_b",
                "dataset",
                "analysis_subset",
                "metric",
            ],
            sort=False,
        )["signed_advantage_branch_a"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_signed_advantage_across_seeds",
                "std": "sd_signed_advantage_across_seeds",
                "min": "minimum_signed_advantage_across_seeds",
                "max": "maximum_signed_advantage_across_seeds",
                "count": "n_seed_pairs",
            }
        )
    )
    atomic_to_csv(
        paired_seed_summary, summary_dir / "paired_metric_summary_across_seeds.csv"
    )

    fit_counts = (
        all_selected.groupby(["branch", "dataset", "seed", "fold"], as_index=False)
        .size()
        .rename(columns={"size": "n_selected_features"})
    )
    atomic_to_csv(fit_counts, summary_dir / "selected_feature_count_by_fit.csv")
    feature_frequency = (
        all_selected.drop_duplicates(["branch", "dataset", "seed", "fold", "feature"])
        .groupby(["branch", "dataset", "feature"], as_index=False)
        .size()
        .rename(columns={"size": "selected_in_outer_fits"})
    )
    feature_frequency["total_outer_fits"] = len(project_config["seeds"]) * int(
        project_config["n_splits"]
    )
    feature_frequency["selection_frequency"] = (
        feature_frequency["selected_in_outer_fits"]
        / feature_frequency["total_outer_fits"]
    )
    feature_frequency = feature_frequency.sort_values(
        ["branch", "dataset", "selection_frequency", "feature"],
        ascending=[True, True, False, True],
    )
    atomic_to_csv(feature_frequency, summary_dir / "feature_selection_frequency.csv")

    jaccard_rows = []
    for (branch, dataset), group in all_selected.groupby(["branch", "dataset"]):
        fits = {
            (int(seed), int(fold)): set(part["feature"].astype(str))
            for (seed, fold), part in group.groupby(["seed", "fold"])
        }
        for (fit_a, features_a), (fit_b, features_b) in combinations(fits.items(), 2):
            union = features_a | features_b
            jaccard_rows.append(
                {
                    "branch": branch,
                    "dataset": dataset,
                    "seed_a": fit_a[0],
                    "fold_a": fit_a[1],
                    "seed_b": fit_b[0],
                    "fold_b": fit_b[1],
                    "jaccard": len(features_a & features_b) / len(union),
                }
            )
    jaccard = pd.DataFrame(jaccard_rows)
    atomic_to_csv(jaccard, summary_dir / "pairwise_feature_jaccard.csv")
    jaccard_summary = (
        jaccard.groupby(["branch", "dataset"])["jaccard"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "n_fit_pairs"})
    )
    atomic_to_csv(jaccard_summary, summary_dir / "feature_jaccard_summary.csv")

    reps = int(
        args.bootstrap_reps
        if args.bootstrap_reps is not None
        else audit_config["bootstrap_reps"]
    )
    if reps < 100:
        raise ValueError("bootstrap-reps must be at least 100")
    bootstrap_seed = int(audit_config["bootstrap_seed"])
    branch_ci_frames = []
    paired_ci_frames = []
    dataset_names = [str(item["name"]) for item in project_config["datasets"]]
    counter = 0
    for dataset in dataset_names:
        for subset in subsets:
            for branch in audit_config["branches"]:
                counter += 1
                frame = subset_frame(
                    mean_oof.query("branch == @branch and dataset == @dataset"),
                    subset,
                    cutoff,
                ).reset_index(drop=True)
                print(
                    f"Bootstrap branch {counter}/{len(dataset_names) * len(subsets) * len(audit_config['branches'])}: "
                    f"{branch}, {dataset}, {subset}",
                    flush=True,
                )
                interval = bootstrap_single_branch(
                    frame, reps, bootstrap_seed + 1000 + counter
                )
                interval.insert(0, "analysis_subset", subset)
                interval.insert(0, "dataset", dataset)
                interval.insert(0, "branch", branch)
                branch_ci_frames.append(interval)

            for comparison_index, (comparison, branch_a, branch_b) in enumerate(
                COMPARISONS, start=1
            ):
                frame_a = subset_frame(
                    mean_oof.query("branch == @branch_a and dataset == @dataset"),
                    subset,
                    cutoff,
                ).reset_index(drop=True)
                frame_b = subset_frame(
                    mean_oof.query("branch == @branch_b and dataset == @dataset"),
                    subset,
                    cutoff,
                ).reset_index(drop=True)
                interval = bootstrap_paired(
                    frame_a,
                    frame_b,
                    reps,
                    bootstrap_seed + 100000 + counter * 10 + comparison_index,
                )
                interval.insert(0, "analysis_subset", subset)
                interval.insert(0, "dataset", dataset)
                interval.insert(0, "branch_b", branch_b)
                interval.insert(0, "branch_a", branch_a)
                interval.insert(0, "comparison", comparison)
                paired_ci_frames.append(interval)
    atomic_to_csv(
        pd.concat(branch_ci_frames, ignore_index=True),
        summary_dir / "branch_metrics_bootstrap_95ci.csv",
    )
    atomic_to_csv(
        pd.concat(paired_ci_frames, ignore_index=True),
        summary_dir / "paired_comparisons_bootstrap_95ci.csv",
    )

    paper_metrics = [
        "pearson_r",
        "spearman_rho",
        "mae",
        "rmse",
        "r2",
        "within_1_twofold_percent",
        "within_2_twofold_percent",
        "median_relative_error_percent",
    ]
    paper = metric_summary.query(
        "analysis_subset == 'all_positive_mic' and metric in @paper_metrics"
    ).pivot(
        index=["branch", "dataset"],
        columns="metric",
        values=["mean_across_seeds", "sd_across_seeds"],
    )
    paper.columns = [f"{metric}_{stat.replace('_across_seeds', '')}" for stat, metric in paper.columns]
    paper = paper.reset_index()
    atomic_to_csv(paper, summary_dir / "paper_table_general_metrics.csv")

    completion = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "run_signature": signature,
        "n_tasks": total,
        "n_prediction_rows": int(len(all_predictions)),
        "n_branches": len(audit_config["branches"]),
        "n_datasets": len(project_config["datasets"]),
        "n_seeds": len(project_config["seeds"]),
        "n_folds": int(project_config["n_splits"]),
        "bootstrap_reps": reps,
        "operational_negative_mic": cutoff,
    }
    atomic_write_json(completion, summary_dir / "AGGREGATION_COMPLETE.json")
    print("LEAKAGE AUDIT AGGREGATION COMPLETE", flush=True)
    print(paper.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
