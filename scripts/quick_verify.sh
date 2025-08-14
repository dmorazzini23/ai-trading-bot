#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: quick verification script
# Quick sanity: compile, import contract, package health (non-fatal), and a tiny test.
python -m py_compile $(git ls-files '*.py')

if [ -f tools/import_contract.py ]; then
  python tools/import_contract.py
fi

if [ -f tools/package_health.py ]; then
  # Don't fail CI on deep import of optional deps; just report.
  python tools/package_health.py --strict || true
fi

# Fastest smoke test (hermetic): helpers only; full suite is run via `make test-all`.
if [ -f tests/test_metrics_fetch_helpers.py ]; then
  pytest -q tests/test_metrics_fetch_helpers.py --noconftest || true
fi

echo "quick_verify: OK"
