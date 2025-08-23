#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: wrapper for repairing stale test imports
dir="artifacts"
mkdir -p "$dir"
python tools/repair_test_imports.py --pkg ai_trading --tests tests --rewrite-map tools/static_import_rewrites.txt --write --report "$dir/import-repair-report.md"
echo "Import repair report written to $dir/import-repair-report.md"
