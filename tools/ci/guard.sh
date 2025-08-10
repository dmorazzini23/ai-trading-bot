#!/usr/bin/env bash
set -euo pipefail
fail=0
git grep -nE '\bexec\s*\(' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true
git grep -nE '(^|[^.])\beval\s*\(' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true
git grep -nE '^\s*except\s*:\s*$' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true
git grep -nE 'yaml\.load\s*\(' -- 'ai_trading/**/*.py' | grep -v 'Loader=' | tee /dev/stderr && fail=1 || true
git grep -nE 'requests\.(get|post|put|delete|patch)\s*\(' -- 'ai_trading/**/*.py' | grep -v 'timeout[[:space:]]*=' | tee /dev/stderr && fail=1 || true
git grep -nE 'subprocess\.(run|Popen|call|check_call|check_output)\s*\(' -- 'ai_trading/**/*.py' | grep -v 'timeout[[:space:]]*=' | tee /dev/stderr && fail=1 || true
git grep -nE '\bdatetime\.now\s*\(\s*\)' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true
git grep -nE '^[[:space:]]*async[[:space:]]+def[\s\S]*^[[:space:]]*time\.sleep\(' -n -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true
git grep -nE 'def[^(]*\([^)]*(\[\]|\{\}|set\(\))' -n -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true
exit $fail