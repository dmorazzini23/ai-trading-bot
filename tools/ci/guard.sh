#!/usr/bin/env bash
set -euo pipefail
fail=0

PKG_DIRS="ai_trading"
EXCLUDE='(venv|\.venv|site-packages|build|dist|migrations|_generated)'

check() {
  local name="$1"; shift
  local pat="$1"; shift
  if find $PKG_DIRS -name "*.py" -exec grep -l "$pat" {} \; | grep -vE "$EXCLUDE" | head -n 1 >/dev/null; then
    echo "FAIL: $name"; 
    find $PKG_DIRS -name "*.py" -exec grep -Hn "$pat" {} \; | grep -vE "$EXCLUDE" | head -n 5
    fail=1
  fi
}

# 1) any exec(
if find $PKG_DIRS -name "*.py" -exec grep -Hn '\bexec\s*(' {} \; | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: exec()"; fail=1
fi

# 2) raw eval( but ignore attribute .eval(
if find $PKG_DIRS -name "*.py" -exec grep -Hn 'eval\s*(' {} \; | grep -v '\.eval\s*(' | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: raw eval()"; fail=1
fi

# 3) bare except:
if find $PKG_DIRS -name "*.py" -exec grep -Hn '^\s*except\s*:\s*$' {} \; | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: bare except"; fail=1
fi

# 4) yaml.load without Loader
if find $PKG_DIRS -name "*.py" -exec grep -Hn 'yaml\.load\s*(' {} \; | grep -v 'Loader=' | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: yaml.load without Loader"; fail=1
fi

# 5) requests.* missing timeout
if find $PKG_DIRS -name "*.py" -exec grep -Hn 'requests\.\(get\|post\|put\|delete\|patch\)\s*(' {} \; | grep -v 'timeout\s*=' | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: requests without timeout"; fail=1
fi

# 6) subprocess.* missing timeout
if find $PKG_DIRS -name "*.py" -exec grep -Hn 'subprocess\.\(run\|Popen\|call\|check_call\|check_output\)\s*(' {} \; | grep -v 'timeout\s*=' | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: subprocess without timeout"; fail=1
fi

# 7) naive datetime.now()
if find $PKG_DIRS -name "*.py" -exec grep -Hn 'datetime\.now\s*(\s*)' {} \; | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: naive datetime.now()"; fail=1
fi

# 8) time.sleep inside async def - this is complex, skip for bash version

# 9) mutable defaults in defs
if find $PKG_DIRS -name "*.py" -exec grep -Hn 'def[^(]*([^)]*\(\[\]\|\{\}\|set()\)' {} \; | grep -vE "$EXCLUDE" | head -n 1; then
  echo "FAIL: mutable default"; fail=1
fi

exit $fail