#!/usr/bin/env bash
set -euo pipefail
found=0
while IFS= read -r -d '' f; do
  if grep -nE "^from[[:space:]]+trade_execution[[:space:]]+import|^import[[:space:]]+trade_execution(\b|,)" "$f" >/dev/null; then
    echo "Legacy import found in $f:"
    grep -nE "^from[[:space:]]+trade_execution[[:space:]]+import|^import[[:space:]]+trade_execution(\b|,)" "$f" || true
    found=1
  fi
done < <(git ls-files '*.py' -z)
exit "$found"