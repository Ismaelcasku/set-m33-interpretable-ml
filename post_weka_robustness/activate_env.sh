#!/usr/bin/env bash
# Source this file: source activate_env.sh
SET_M33_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SET_M33_PROJECT_ROOT/.venv/bin/python" ]]; then
  source "$SET_M33_PROJECT_ROOT/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SET_M33_ENV_NAME:-setm33-robustness}"
else
  echo "No .venv or conda environment found. Run setup_environment.sh first." >&2
  return 1 2>/dev/null || exit 1
fi
export PYTHONPATH="$SET_M33_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset SET_M33_PROJECT_ROOT

