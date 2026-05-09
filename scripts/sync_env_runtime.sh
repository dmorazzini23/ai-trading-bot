#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  else
    echo "python3.12 is required; run bash scripts/bootstrap.sh first" >&2
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "PYTHON_BIN must point to a Python 3.12 interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

ENV_SRC="${AI_TRADING_ENV_SRC:-.env}"
if [[ ! -f "${ENV_SRC}" ]]; then
  echo "AI_TRADING_ENV_SRC does not exist: ${ENV_SRC}" >&2
  exit 1
fi
if [[ ! -r "${ENV_SRC}" ]]; then
  echo "AI_TRADING_ENV_SRC is not readable: ${ENV_SRC}" >&2
  exit 1
fi
PACKAGED_RUNTIME_DIR="${AI_TRADING_PACKAGED_RUNTIME_DIR:-/run/ai-trading-bot}"
if [[ -n "${AI_TRADING_RUNTIME_ENV_DST:-}" ]]; then
  RUNTIME_ENV_DST="${AI_TRADING_RUNTIME_ENV_DST}"
elif [[ -d "${PACKAGED_RUNTIME_DIR}" ]]; then
  RUNTIME_ENV_DST="${PACKAGED_RUNTIME_DIR}/ai-trading-runtime.env"
else
  RUNTIME_ENV_DST="runtime/ai-trading-runtime.env"
fi
RUNTIME_ENV_DIR="$(dirname "${RUNTIME_ENV_DST}")"
case "${RUNTIME_ENV_DST}" in
  ""|"/"|".env"|"${ENV_SRC}")
    echo "refusing unsafe runtime env destination: ${RUNTIME_ENV_DST}" >&2
    exit 1
    ;;
esac
case "${RUNTIME_ENV_DST}" in
  "${PACKAGED_RUNTIME_DIR}"/*)
    if [[ ! -d "${RUNTIME_ENV_DIR}" ]]; then
      echo "packaged runtime env directory is missing: ${RUNTIME_ENV_DIR}" >&2
      exit 1
    fi
    if [[ ! -w "${RUNTIME_ENV_DIR}" ]]; then
      echo "packaged runtime env directory is not writable: ${RUNTIME_ENV_DIR}" >&2
      exit 1
    fi
    ;;
  *)
    mkdir -p "${RUNTIME_ENV_DIR}"
    ;;
esac

"${PYTHON_BIN}" scripts/runtime_env_sync.py --src "${ENV_SRC}" --dst "${RUNTIME_ENV_DST}"
