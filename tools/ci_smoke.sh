#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: stabilized smoke pipeline (Python lint, optional shellcheck, targeted tests)

# Install dev test dependencies unless explicitly skipped. This ensures
# `pytest -n` works (xdist present) and avoids ModuleNotFoundError for psutil.
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  if [ -f "requirements/dev.txt" ]; then
    python -m pip install --upgrade pip >/dev/null 2>&1 || true
    python -m pip install -r requirements/dev.txt
  fi
fi

# -----------------
# Python lint (ruff)
# -----------------
# Lint only Python files; avoid non-Python paths that ruff can't parse.
if python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('ruff') else 1)
PY
then
  # Prefer git for speed; fall back to find(1) if outside a repo
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    PY_FILES=$(git ls-files '*.py' || true)
  else
    PY_FILES=$(find . -type f -name '*.py' -not -path './venv/*' || true)
  fi
  if [ -n "${PY_FILES}" ]; then
    python -m ruff check ${PY_FILES}
  else
    echo "[ci_smoke] No Python files found for ruff."
  fi
else
  echo "[ci_smoke] ruff not installed; skipping Python lint."
fi

# ----------------------
# Optional shell linting
# ----------------------
if command -v shellcheck >/dev/null 2>&1; then
  if [ -f tools/ci_smoke.sh ]; then
    shellcheck tools/ci_smoke.sh || true
  fi
else
  echo "[ci_smoke] shellcheck not installed; skipping shell lint."
fi

# -----------------
# Targeted smoke run
# -----------------
# Run only the tiny smoke tests via the hardened runner with autoload OFF.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}
python tools/run_pytest.py --disable-warnings -k "runner_smoke or utils_timing" -q

echo "[ci_smoke] Completed."

