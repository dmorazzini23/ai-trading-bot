#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime + dev dependencies quickly
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  else
    echo "[bootstrap] python3.12 is required to create ${ROOT_DIR}/venv" >&2
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "[bootstrap] PYTHON_BIN must point to a Python 3.12 interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -x "${VENV_PYTHON}" ]] && ! "${VENV_PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "[bootstrap] recreating stale virtualenv with Python 3.12" >&2
  rm -rf "$ROOT_DIR/venv"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  "${PYTHON_BIN}" -m venv "$ROOT_DIR/venv" --system-site-packages
fi

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r "$ROOT_DIR/requirements-dev.txt" -r "$ROOT_DIR/requirements-test.txt"
"${VENV_PYTHON}" -m pip install -e "$ROOT_DIR" --no-deps
PYTHON_BIN="${VENV_PYTHON}" bash "$ROOT_DIR/ci/scripts/verify_alpaca_sdk.sh"

echo "[bootstrap] dependencies installed" >&2
