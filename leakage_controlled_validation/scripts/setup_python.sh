#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_command="${PYTHON_COMMAND:-python3}"

"$python_command" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit(
        f"Python 3.9 or newer is required; found {sys.version.split()[0]}. "
        "Load a newer Python module or create a conda environment first."
    )
print("Using Python", sys.version.split()[0])
PY

"$python_command" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python - <<'PY'
import numpy, pandas, scipy, sklearn
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("scikit-learn", sklearn.__version__)
PY
