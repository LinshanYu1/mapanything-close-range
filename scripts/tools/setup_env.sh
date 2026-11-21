#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BASE_PYTHON="${PYTHON312_BIN:-python3.12}"
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
if [ ! -e .venv ]; then
  "$BASE_PYTHON" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv is not Python 3.12; select a different environment first"'
.venv/bin/python -m pip install -r requirements-macos-py312.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/tools/check_environment.py
