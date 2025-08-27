#!/usr/bin/env bash
# Run fast sanity checks: compilation, lint, type-check, and tests.
# Fails on usage of legacy imports or alpaca-trade-api.
set -euo pipefail

ci/scripts/forbid_alpaca_trade_api.sh
ci/scripts/forbid_legacy_imports.sh

python - <<'PY'
import subprocess, sys, pathlib
files = [str(p) for p in pathlib.Path('.').rglob('*.py')]
subprocess.check_call([sys.executable, '-m', 'py_compile', *files])
PY

ruff check .
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mypy ai_trading tests
pytest -q
