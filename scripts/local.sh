#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OLD_PROJECT=/Users/linshanyu/Desktop/dl-project
if [ -n "${PYTHON_BIN:-}" ]; then
  PILOT_PYTHON="$PYTHON_BIN"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PILOT_PYTHON="$ROOT/.venv/bin/python"
else
  PILOT_PYTHON="$OLD_PROJECT/.venv-baseline/bin/python"
fi
if [ ! -x "$PILOT_PYTHON" ]; then
  echo 'Set PYTHON_BIN to a Python environment with the MapAnything dependencies installed.' >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT/third_party/map-anything${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$ROOT/outputs/.matplotlib"
export HF_HOME="${HF_HOME:-$OLD_PROJECT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$OLD_PROJECT/.cache/torch}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ $# -eq 0 ]; then
  echo 'Usage: bash scripts/local.sh scripts/phone/annotate.py'
  exit 1
fi
exec "$PILOT_PYTHON" "$@"
