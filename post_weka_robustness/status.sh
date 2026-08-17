#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${SET_M33_SUBMISSION_DIR:-$ROOT/results/submission}/job_ids.tsv"
if [[ ! -f "$JOBS" ]]; then
  echo "No submission record at $JOBS" >&2
  exit 1
fi
IDS="$(tail -n +2 "$JOBS" | cut -f2 | paste -sd, -)"
echo "Active jobs:"
squeue -j "$IDS" -o '%.18i %.22j %.10T %.12M %.6D %.6C %R' || true
echo
echo "Accounting summary:"
sacct -j "$IDS" --format=JobID,JobName%24,State,Elapsed,AllocCPUS,MaxRSS,ExitCode -X || true
