#!/usr/bin/env bash
# AI-AGENT-REF: wrapper for import repair utility
set -euo pipefail
PKG="${1:-ai_trading}"
TESTS_DIR="${2:-tests}"
ARTIFACT="${3:-artifacts/import-repair-report.md}"

python tools/repair_test_imports.py --pkg "$PKG" --tests "$TESTS_DIR" --write --report "$ARTIFACT"
echo "Import repair report written to $ARTIFACT"

