#!/usr/bin/env bash
set -euo pipefail
fail=0

# 1) exec is always unsafe
git grep -nE '\bexec\s*\(' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true

# 2) raw eval( but ignore attribute .eval(  (e.g., model.eval())
git grep -nE '(^|[^.])\beval\s*\(' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true

# 3) bare except
git grep -nE '^[[:space:]]*except[[:space:]]*:[[:space:]]*$' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true

# 4) yaml.load without Loader
git grep -nE 'yaml\.load\s*\(' -- 'ai_trading/**/*.py' | grep -v 'Loader=' | tee /dev/stderr && fail=1 || true

# 5) requests.* without timeout
git grep -nE 'requests\.(get|post|put|delete|patch)\s*\(' -- 'ai_trading/**/*.py' | grep -v 'timeout[[:space:]]*=' | tee /dev/stderr && fail=1 || true

# 6) subprocess.* without timeout
git grep -nE 'subprocess\.(run|Popen|call|check_call|check_output)\s*\(' -- 'ai_trading/**/*.py' | grep -v 'timeout[[:space:]]*=' | tee /dev/stderr && fail=1 || true

# 7) naive datetime.now()
git grep -nE '\bdatetime\.now\s*\(\s*\)' -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true

# 8) time.sleep inside async def (coarse but good signal)
git grep -nE '^[[:space:]]*async[[:space:]]+def[\s\S]*^[[:space:]]*time\.sleep\(' -n -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true

# 9) mutable defaults in defs (coarse)
git grep -nE 'def[^(]*\([^)]*(\[\]|\{\}|set\(\))' -n -- 'ai_trading/**/*.py' | tee /dev/stderr && fail=1 || true

exit $fail