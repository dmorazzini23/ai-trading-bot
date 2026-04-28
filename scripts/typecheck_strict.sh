#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./venv/bin/python" ]]; then
    PYTHON_BIN="./venv/bin/python"
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

# High-signal strict type checks for critical runtime paths.
if [[ -z "${MYPY_CACHE_DIR:-}" ]]; then
  if [[ -w ".mypy_cache" ]] || [[ ! -e ".mypy_cache" && -w "." ]]; then
    export MYPY_CACHE_DIR=".mypy_cache"
  else
    export MYPY_CACHE_DIR="/tmp/mypy_cache_ai_trading"
  fi
fi

# Broad baseline coverage for the application tree.
"${PYTHON_BIN}" -m mypy --config-file mypy.ini ai_trading

"${PYTHON_BIN}" -m mypy --config-file mypy_strict.ini \
  ai_trading/config/management.py \
  ai_trading/config/alpaca.py \
  ai_trading/config/__init__.py \
  ai_trading/config/settings.py \
  ai_trading/config/runtime.py \
  ai_trading/validation/validate_env.py \
  ai_trading/validation/require_env.py \
  ai_trading/health_payload.py \
  ai_trading/health.py \
  ai_trading/logging/emit_once.py \
  ai_trading/strategy_allocator.py \
  ai_trading/rl/module.py \
  scripts/health_check.py \
  scripts/production_monitoring.py

# Runtime-critical modules that are not yet strict-clean still get explicit
# baseline type coverage in this gate.
"${PYTHON_BIN}" -m mypy --config-file mypy.ini \
  ai_trading/__main__.py \
  ai_trading/alpaca_api.py \
  ai_trading/app.py \
  ai_trading/core/bot_engine.py \
  ai_trading/data/fallback/concurrency.py \
  ai_trading/data/fetch/fallback_concurrency.py \
  ai_trading/data/universe.py \
  ai_trading/data/provider_monitor.py \
  ai_trading/execution/engine.py \
  ai_trading/execution/live_trading.py \
  ai_trading/settings.py \
  ai_trading/main.py \
  ai_trading/model_loader.py \
  ai_trading/logging/__init__.py \
  ai_trading/logging_filters.py \
  ai_trading/http/pooling.py \
  ai_trading/paths.py \
  ai_trading/net/http.py \
  ai_trading/data/fetch/__init__.py \
  ai_trading/policy/compiler.py \
  ai_trading/utils/env.py \
  ai_trading/utils/environment.py \
  ai_trading/utils/exec.py \
  ai_trading/env/__init__.py \
  ai_trading/util/env_check.py
