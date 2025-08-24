#!/usr/bin/env bash
set -euo pipefail

# Stabilized smoke pipeline: non-blocking ruff; explicit test targets; plugin autoload off.

# Install dev test dependencies unless explicitly skipped (xdist, psutil, ruff, etc.).
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  if [ -f "requirements/dev.txt" ]; then
    python -m pip install --upgrade pip >/dev/null 2>&1 || true
    python -m pip install -r requirements/dev.txt
  fi
fi

# -----------------
# Python lint (ruff) — non-blocking in smoke
# -----------------
if command -v ruff >/dev/null 2>&1; then
  echo "[ci_smoke] Ruff lint (non-blocking)"
  set +e
  ruff check .
  set -e
else
  echo "[ci_smoke] Ruff not found; skipping"
fi

# -----------------
# Targeted smoke run
# -----------------
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}
echo "[ci_smoke] Running minimal smoke suite (3 files)"
python tools/run_pytest.py --disable-warnings -q \
  tests/test_runner_smoke.py \
  tests/test_utils_timing.py \
  tests/test_trading_config_aliases.py

echo "[ci_smoke] Completed."

