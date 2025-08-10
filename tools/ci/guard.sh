#!/usr/bin/env bash
set -euo pipefail
fail=0
echo "Guard checks…"

# disallow eval/exec (but allow .eval() method calls)
if git grep -nE '\b(eval|exec)\s*\(' -- 'ai_trading/**/*.py' | grep -v '\.eval\s*(' | tee /dev/stderr; then fail=1; fi

# disallow bare except
if git grep -nE '^\s*except\s*:\s*$' -- 'ai_trading/**/*.py' | tee /dev/stderr; then fail=1; fi

# disallow yaml.load without loader
if git grep -nE 'yaml\.load\s*\(' -- 'ai_trading/**/*.py' | grep -v 'Loader=' | tee /dev/stderr; then fail=1; fi

# enforce requests timeout
request_lines=$(git grep -nE 'requests\.(get|post|put|delete|patch)\s*\(' -- 'ai_trading/**/*.py' || true)
if [ -n "$request_lines" ]; then
    echo "$request_lines" | while read line; do
        file=$(echo "$line" | cut -d':' -f1)
        lineno=$(echo "$line" | cut -d':' -f2)
        # Check the current line and next 5 lines for timeout
        if ! sed -n "${lineno},$((lineno+5))p" "$file" | grep -q 'timeout='; then
            echo "$line" >&2
            exit 1
        fi
    done || fail=1
fi

# subprocess hygiene
subprocess_lines=$(git grep -nE 'subprocess\.(run|Popen|call|check_call|check_output)\s*\(' -- 'ai_trading/**/*.py' || true)
if [ -n "$subprocess_lines" ]; then
    echo "$subprocess_lines" | while read line; do
        file=$(echo "$line" | cut -d':' -f1)
        lineno=$(echo "$line" | cut -d':' -f2)
        # Check the current line and next 5 lines for timeout
        if ! sed -n "${lineno},$((lineno+5))p" "$file" | grep -q 'timeout='; then
            echo "$line" >&2
            exit 1
        fi
    done || fail=1
fi

# naive datetime.now
if git grep -nE '\bdatetime\.now\s*\(\s*\)' -- 'ai_trading/**/*.py' | tee /dev/stderr; then fail=1; fi

# wildcard imports in package code
if git grep -nE 'from\s+[.\w]+\s+import\s+\*' -- 'ai_trading/**/*.py' | tee /dev/stderr; then fail=1; fi

# path literals to data/
if git grep -nE '["\x27](\./)?data/' -- 'ai_trading/**/*.py' | tee /dev/stderr; then fail=1; fi

exit $fail