#!/usr/bin/env bash
set -euo pipefail
fail=0

INCLUDE='ai_trading/**/*.py'
EXCLUDE='(venv|\.venv|site-packages|build|dist|migrations|_generated)'

echo "Checking for shim patterns in ai_trading/..."

# 1) Import fallbacks (try/except ImportError around import/from)
echo "1. Checking for import fallbacks..."
import_guards=$(git grep -nE 'try:[[:space:]]*$' -- ai_trading/ | \
    while IFS= read -r line; do
        file=$(echo "$line" | cut -d: -f1)
        linenum=$(echo "$line" | cut -d: -f2)
        # Check next few lines for import and except ImportError
        awk -v start="$linenum" 'NR>=start && NR<=start+6 {
            if ($0 ~ /(^[[:space:]]*import|^[[:space:]]*from[[:space:]]+.*[[:space:]]+import)/) import_found=1
            if ($0 ~ /^[[:space:]]*except[[:space:]]+ImportError/) except_found=1
        } END {
            if (import_found && except_found) exit 0; else exit 1
        }' "$file" && echo "$line"
    done) || true

if [ -n "$import_guards" ]; then
    echo "Found import guard patterns:"
    echo "$import_guards"
    fail=1
fi

# 2) Config magic
echo "2. Checking for config magic..."
config_getattr=$(git grep -nE 'def[[:space:]]+__getattr__\s*\(' -- ai_trading/config/ || true)
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