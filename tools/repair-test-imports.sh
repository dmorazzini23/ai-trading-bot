#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/artifacts"
python "${ROOT}/tools/repair_test_imports.py" --pkg ai_trading --tests tests \
  --rewrite-map "${ROOT}/tools/static_import_rewrites.txt" \
  --write --report "${ROOT}/artifacts/import-repair-report.md"
echo "Wrote ${ROOT}/artifacts/import-repair-report.md"
# AI-AGENT-REF: lightweight wrapper for import repair

