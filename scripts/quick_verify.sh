#!/usr/bin/env bash
# Run fast sanity checks: compilation, lint, type-check, and tests.
# Fails if unsupported Alpaca SDK versions are installed or legacy imports exist.
set -euo pipefail

ci/scripts/verify_alpaca_sdk.sh
ci/scripts/forbid_legacy_imports.sh

python3 - <<'PY'
import subprocess, sys, pathlib
files = [str(p) for p in pathlib.Path('.').rglob('*.py')]
subprocess.check_call([sys.executable, '-m', 'py_compile', *files])
PY

ruff check .
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mypy ai_trading tests
scripts/typecheck_strict.sh
python3 tools/ci/guard_runtime_env_access.py
pytest -q
