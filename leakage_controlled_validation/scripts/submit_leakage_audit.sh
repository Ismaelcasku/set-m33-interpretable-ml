#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PROJECT_PYTHON:-$project_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found: $python_bin" >&2
  echo "Create .venv first or export PROJECT_PYTHON=/absolute/path/to/python." >&2
  exit 1
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 --smoke | --full | --retry | --aggregate-only" >&2
  exit 2
fi
mode="$1"
mkdir -p logs results/leakage_audit_v1

if [[ "$mode" == "--aggregate-only" ]]; then
  aggregate_job="$(sbatch --parsable slurm/aggregate_leakage.slurm)"
  aggregate_job="${aggregate_job%%;*}"
  echo "Aggregation job: $aggregate_job"
  exit 0
fi

"$python_bin" scripts/preflight_leakage_audit.py --project-root "$project_root"

branches=(legacy_global nested_matched)
offsets=(0 550)

if [[ "$mode" == "--retry" ]]; then
  "$python_bin" scripts/validate_global_selection.py --project-root "$project_root"
  retry_jobs=()
  for index in "${!branches[@]}"; do
    branch="${branches[$index]}"
    offset="${offsets[$index]}"
    missing_spec="$("$python_bin" scripts/check_leakage_progress.py \
      --project-root "$project_root" --missing-array --branch "$branch")"
    if [[ -z "$missing_spec" ]]; then
      continue
    fi
    job_id="$(sbatch --parsable \
      --export="ALL,AUDIT_TASK_OFFSET=${offset}" \
      --array="${missing_spec}%20" slurm/run_leakage_array.slurm)"
    job_id="${job_id%%;*}"
    retry_jobs+=("$job_id")
    echo "Retry $branch: $job_id"
  done
  if [[ ${#retry_jobs[@]} -eq 0 ]]; then
    echo "No missing fold tasks; submit aggregation with $0 --aggregate-only"
    exit 0
  fi
  dependency="$(IFS=:; echo "${retry_jobs[*]}")"
  aggregate_job="$(sbatch --parsable --dependency="afterok:${dependency}" slurm/aggregate_leakage.slurm)"
  aggregate_job="${aggregate_job%%;*}"
  echo "Dependent aggregation job: $aggregate_job"
  exit 0
fi

if [[ "$mode" != "--smoke" && "$mode" != "--full" ]]; then
  echo "Unknown mode: $mode" >&2
  exit 2
fi

global_job="$(sbatch --parsable slurm/run_global_selection.slurm)"
global_job="${global_job%%;*}"
validation_job="$(sbatch --parsable --dependency="afterok:${global_job}" slurm/validate_global_selection.slurm)"
validation_job="${validation_job%%;*}"
if [[ "$mode" == "--smoke" ]]; then
  smoke_jobs=()
  for index in "${!branches[@]}"; do
    branch="${branches[$index]}"
    offset="${offsets[$index]}"
    job_id="$(sbatch --parsable \
      --dependency="afterok:${validation_job}" \
      --export="ALL,AUDIT_TASK_OFFSET=${offset}" \
      --array=0 slurm/run_leakage_array.slurm)"
    job_id="${job_id%%;*}"
    smoke_jobs+=("$job_id")
    echo "Smoke $branch: $job_id"
  done
  echo "Global-pool array: $global_job"
  echo "Pool validation: $validation_job"
  echo "After completion, inspect logs and run scripts/check_leakage_progress.py."
else
  fold_jobs=()
  for index in "${!branches[@]}"; do
    branch="${branches[$index]}"
    offset="${offsets[$index]}"
    job_id="$(sbatch --parsable \
      --dependency="afterok:${validation_job}" \
      --export="ALL,AUDIT_TASK_OFFSET=${offset}" \
      slurm/run_leakage_array.slurm)"
    job_id="${job_id%%;*}"
    fold_jobs+=("$job_id")
    echo "Full array $branch: $job_id"
  done
  dependency="$(IFS=:; echo "${fold_jobs[*]}")"
  aggregate_job="$(sbatch --parsable --dependency="afterok:${dependency}" slurm/aggregate_leakage.slurm)"
  aggregate_job="${aggregate_job%%;*}"
  echo "Global-pool array: $global_job"
  echo "Pool validation: $validation_job"
  echo "Dependent aggregation job: $aggregate_job"
fi
