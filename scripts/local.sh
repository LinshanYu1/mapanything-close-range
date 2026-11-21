#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -n "${PYTHON_BIN:-}" ]; then
  PILOT_PYTHON="$PYTHON_BIN"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PILOT_PYTHON="$ROOT/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PILOT_PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PILOT_PYTHON="$CONDA_PREFIX/bin/python"
else
  PILOT_PYTHON="$ROOT/.venv-baseline/bin/python"
fi
if [[ "$PILOT_PYTHON" != */* ]]; then
  PILOT_PYTHON="$(command -v "$PILOT_PYTHON" || true)"
fi
if [ ! -x "$PILOT_PYTHON" ]; then
  echo 'No usable Python found. Activate your virtual environment, or set PYTHON_BIN to its Python executable.' >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT/third_party/map-anything${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$ROOT/outputs/.matplotlib"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ $# -eq 0 ]; then
  echo 'Usage: bash scripts/local.sh scripts/phone/annotate.py'
  exit 1
fi
exec "$PILOT_PYTHON" "$@"
