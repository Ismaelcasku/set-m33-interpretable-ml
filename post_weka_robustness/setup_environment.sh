#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---conda}"
if [[ "$MODE" == "--conda" ]]; then
  if command -v mamba >/dev/null 2>&1; then
    mamba env update -f "$ROOT/environment.yml" --prune
  elif command -v conda >/dev/null 2>&1; then
    conda env update -f "$ROOT/environment.yml" --prune
  else
    echo "conda/mamba not found; use: bash setup_environment.sh --venv" >&2
    exit 1
  fi
elif [[ "$MODE" == "--venv" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip wheel
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
else
  echo "Usage: bash setup_environment.sh [--conda|--venv]" >&2
  exit 2
fi
echo "Environment ready. Run: source activate_env.sh"
