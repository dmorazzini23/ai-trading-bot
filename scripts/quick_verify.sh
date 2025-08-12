#!/usr/bin/env sh
set -euo pipefail

echo "== Shim checks =="
grep -R --line-number "_execute_sliced" ai_trading/execution/engine.py || echo "OK: _execute_sliced gone"
grep -R --line-number "prepare_indicators_compat" ai_trading/core/bot_engine.py || echo "OK: prepare_indicators_compat gone"

echo "== ImportError guards (prod) =="
if grep -n "ImportError" ai_trading/execution/transaction_costs.py ai_trading/risk/engine.py; then
  echo "FAIL: ImportError guard left"
  exit 1
else
  echo "OK: no ImportError guards"
fi

echo "== Broad catches in hot paths =="
if grep -n "except Exception" ai_trading/core/bot_engine.py | grep -E "_load_primary_model|screen_(candidates|universe)|check_market_regime|detect_regime_state|HEALTH|submit|cancel"; then
  echo "FAIL: broad catch remained in core hot paths"
  exit 1
else
  echo "OK: core hot paths narrowed"
fi
if grep -n "except Exception" ai_trading/execution/production_engine.py ai_trading/execution/live_trading.py | grep -E "submit|cancel"; then
  echo "FAIL: broad catch remained at submit/cancel"
  exit 1
else
  echo "OK: submit/cancel narrowed"
fi

echo "== Compile =="
python - <<'PY'
import sys, subprocess, shlex
files = subprocess.check_output(shlex.split("git ls-files '*.py'")).decode().splitlines()
import py_compile
for f in files:
    py_compile.compile(f, doraise=True)
print("OK: py_compile passed for", len(files), "files")
PY
