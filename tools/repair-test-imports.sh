#!/usr/bin/env bash
set -euo pipefail
PKG="${1:-ai_trading}"
TESTS="${2:-tests}"
ART="artifacts"
REPORT="$ART/import-repair-report.md"

mkdir -p "$ART"

python tools/repair_test_imports.py \
  --pkg "$PKG" \
  --tests "$TESTS" \
  --write \
  --report "$REPORT"

echo "Report written to $REPORT"
