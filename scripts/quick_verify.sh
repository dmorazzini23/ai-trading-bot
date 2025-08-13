#!/usr/bin/env sh
set -euo pipefail


# Ensure dev deps (pytest, xdist, pydantic) are available locally
if [ -f requirements-dev.txt ]; then
  pip install -r requirements-dev.txt >/dev/null
fi

echo "[verify] Python version:"
python -V

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

echo "== Broad catches in hot paths (core) =="
if grep -n "except Exception" ai_trading/core/bot_engine.py | grep -E "_load_primary_model|screen_(candidates|universe)|check_market_regime|detect_regime_state|HEALTH|submit|cancel"; then
  echo "FAIL: broad catch remained in core hot paths"
  exit 1
else
  echo "OK: core hot paths narrowed"
fi

echo "== Broad catches at submit/cancel (execution) =="
if grep -n "except Exception" ai_trading/execution/production_engine.py ai_trading/execution/live_trading.py | grep -E "submit|cancel"; then
  echo "FAIL: broad catch remained at submit/cancel"
  exit 1
else
  echo "OK: submit/cancel narrowed"
fi

echo "== Compile =="
python - <<'PY'
import subprocess, shlex, py_compile
files = subprocess.check_output(shlex.split("git ls-files '*.py'" )).decode().splitlines()
for f in files:
    py_compile.compile(f, doraise=True)
print("OK: py_compile passed for", len(files), "files")
PY

echo "== Import sanity =="
python - <<'PY'
import importlib

for mod in [
    "numpy",
    "pandas",
    "pandas_ta",
    "pandas_market_calendars",
    "alpaca_trade_api",
    "cachetools",
    "psutil",
    "ai_trading.execution.slippage",
    "ai_trading.utils.memory_optimizer",
]:
    importlib.import_module(mod)
print("ok")
PY

python - <<'PY'
import pydantic, sys
print("pydantic:", pydantic.__version__)
try:
    import pydantic_settings
    print("pydantic_settings: ok")
except Exception as e:
    print("pydantic_settings import failed:", type(e).__name__, e)
    sys.exit(1)
PY

python - <<'PY'
from ai_trading.config import get_alpaca_config
c = get_alpaca_config()
print("alpaca_cfg:", bool(c.base_url and c.key_id))
PY

