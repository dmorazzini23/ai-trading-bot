#!/usr/bin/env bash

# Fail the build if deprecated root-level modules are imported. These modules
# previously existed at the repository root but have been migrated into the
# `ai_trading` package. Importing them directly (e.g. `from signals import`)
# would bypass the new package structure.

set -euo pipefail

legacy_modules=(trade_execution signals portfolio rebalancer data_fetcher pipeline indicators)
found=0

while IFS= read -r -d '' f; do
  for mod in "${legacy_modules[@]}"; do
    pattern="^from[[:space:]]+${mod}[[:space:]]+import|^import[[:space:]]+${mod}(\\b|,)"
    if grep -nE "$pattern" "$f" >/dev/null; then
      echo "Legacy import '$mod' found in $f:"
      grep -nE "$pattern" "$f" || true
      found=1
    fi
  done
done < <(git ls-files '*.py' -z)

exit "$found"
