#!/usr/bin/env bash
set -euo pipefail
fail=0

EXCLUDE='(venv|\.venv|site-packages|build|dist|migrations|_generated)'
ALPACA_RUNTIME_PATHS=(
  ai_trading/alpaca_api.py
  ai_trading/core/alpaca_client.py
  ai_trading/execution/live_trading.py
  ai_trading/broker/adapters.py
)

echo "Checking for shim patterns in ai_trading/..."

# 1) Alpaca runtime compatibility surfaces
echo "1. Checking Alpaca runtime compatibility surfaces..."
runtime_shim_markers=$(git grep -nE '(_HTTPShim|TradingClientAdapter|CompatTradingClient|compatibility layer|shim)' -- "${ALPACA_RUNTIME_PATHS[@]}" || true)
if [ -n "$runtime_shim_markers" ]; then
    echo "Found Alpaca runtime shim markers:"
    echo "$runtime_shim_markers"
    fail=1
fi

legacy_method_mutations=$(git grep -nE 'setattr\([^,]+,[[:space:]]*"(list_orders|cancel_order|get_order|list_positions)"' -- "${ALPACA_RUNTIME_PATHS[@]}" || true)
if [ -n "$legacy_method_mutations" ]; then
    echo "Found legacy Alpaca method mutation:"
    echo "$legacy_method_mutations"
    fail=1
fi

legacy_submit_kwargs=$(git grep -nE 'submit_order\(\*\*|submit_order\(dict\(' -- "${ALPACA_RUNTIME_PATHS[@]}" || true)
if [ -n "$legacy_submit_kwargs" ]; then
    echo "Found legacy Alpaca submit_order call style:"
    echo "$legacy_submit_kwargs"
    fail=1
fi

shim_files=$(find ai_trading -type f \( -iname '*shim*.py' -o -iname '*compat*.py' \) \
    ! -path '*/__pycache__/*' | sort || true)
if [ -n "$shim_files" ]; then
    echo "Found runtime shim/compat files:"
    echo "$shim_files"
    fail=1
fi

# 2) Config magic
echo "2. Checking for config magic..."
config_getattr=$(git grep -nE 'def[[:space:]]+__getattr__\s*\(' -- ai_trading/config/ | \
    grep -vE '^ai_trading/config/runtime.py:[0-9]+:[[:space:]]+def[[:space:]]+__getattr__\s*\(' || true)
if [ -n "$config_getattr" ]; then
    echo "Found __getattr__ functions in config:"
    echo "$config_getattr"
    fail=1
fi

uppercase_props=$(git grep -nE '@property[[:space:]]+def[[:space:]]+[A-Z0-9_]+\s*\(' -- ai_trading/config/ || true)
if [ -n "$uppercase_props" ]; then
    echo "Found uppercase alias properties in config:"
    echo "$uppercase_props"
    fail=1
fi

# 3) Runtime mock classes
echo "3. Checking for runtime mock classes..."
mock_classes=$(git grep -nE 'class[[:space:]]+Mock[A-Za-z0-9_]+' -- ai_trading/ || true)
if [ -n "$mock_classes" ]; then
    echo "Found Mock classes in runtime code:"
    echo "$mock_classes"
    fail=1
fi

# 4) Dynamic exec/eval (not attribute .eval())
echo "4. Checking for dynamic exec/eval..."
exec_calls=$(git grep -nE '\bexec\s*\(' -- ai_trading/ || true)
if [ -n "$exec_calls" ]; then
    echo "Found exec() calls:"
    echo "$exec_calls"
    fail=1
fi

eval_calls=$(git grep -nE '(^|[^.])\beval\s*\(' -- ai_trading/ || true)
if [ -n "$eval_calls" ]; then
    echo "Found eval() calls:"
    echo "$eval_calls"
    fail=1
fi

if [ $fail -eq 0 ]; then
    echo "✅ No shim patterns detected"
else
    echo "❌ Shim patterns found - see above"
fi

exit $fail
