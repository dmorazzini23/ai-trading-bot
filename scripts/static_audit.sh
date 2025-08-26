#!/usr/bin/env bash
set -euo pipefail

echo "== Compile =="
python -m py_compile $(git ls-files '*.py')

echo "== Broad except Exception (top offenders) =="
grep -R --line-number --include='*.py' "except Exception" ai_trading | \
  awk -F: '{c[$1]++} END{for (f in c) printf "%5d  %s\n", c[f], f}' | sort -nr || true

echo "== Shims =="
grep -R --line-number "_execute_sliced\|prepare_indicators_compat" ai_trading || echo "OK: no shims"

echo "== ImportError guards in prod =="
grep -R --line-number "except ImportError" ai_trading || echo "OK: no ImportError guards"

echo "== ATR import wiring =="
if grep -R --line-number "from ai_trading\.risk\.engine import compute_atr\|risk\.compute_atr" ai_trading; then
  echo "FAIL: compute_atr imported from risk.engine (should be ai_trading.indicators)"; exit 1
else
  echo "OK: no stray compute_atr imports from risk.engine"
fi

echo "== Logger hygiene (disallow kwargs other than extra=) =="
bad_logs=$(grep -R --line-number --include='*.py' "_log\.\(info\|warning\|error\|debug\)\([^)]*=[^)]*\)" ai_trading | grep -v "extra=" || true)
if [[ -n "${bad_logs}" ]]; then
  echo "WARN: logger calls with unexpected kwargs:"
  echo "${bad_logs}"
else
  echo "OK: logger kwargs look clean"
fi

echo "== Legacy ctx uses (not alias/param) =="
grep -R --line-number --include='*.py' "\bctx\b" ai_trading | \
  grep -v "ctx = runtime" | grep -v "def .*ctx" || echo "OK: no legacy ctx uses"

echo "== Model/trade-log path hazards =="
if grep -R --line-number --include='*.py' "abspath( *None" ai_trading; then
  echo "FAIL: abspath(None) detected"; exit 1
else
  echo "OK: abspath(None) not found"
fi
grep -R --line-number --include='*.py' "os\.path\.exists( *None" ai_trading && { echo "FAIL: exists(None)"; exit 1; } || echo "OK: no exists(None)"

echo "== Heavy ML/RL top-level imports =="
grep -R --line-number --include='*.py' "^import torch\|^from torch" ai_trading | grep -v "TYPE_CHECKING" || true
grep -R --line-number --include='*.py' "^import stable_baselines3\|^from stable_baselines3" ai_trading | grep -v "TYPE_CHECKING" || true

echo "== Orphan/typo’d modules =="
echo "All static checks completed."
