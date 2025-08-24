#!/usr/bin/env bash
set -euo pipefail

: "${TOP_N:=5}"
: "${FAIL_ON_IMPORT_ERRORS:=0}"
: "${DISABLE_ENV_ASSERT:=0}"
: "${SKIP_INSTALL:=0}"
: "${IMPORT_REPAIR_REPORT:=artifacts/import-repair-report.md}"

# AI-AGENT-REF: ensure dev deps present for raw pytest use
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  if [ -f "requirements/dev.txt" ]; then
    python -m pip install --upgrade pip >/dev/null 2>&1 || true
    python -m pip install -r requirements/dev.txt
  fi
fi

mkdir -p "$(dirname "$IMPORT_REPAIR_REPORT")"

TOP_N="$TOP_N" \
FAIL_ON_IMPORT_ERRORS="$FAIL_ON_IMPORT_ERRORS" \
DISABLE_ENV_ASSERT="$DISABLE_ENV_ASSERT" \
SKIP_INSTALL="$SKIP_INSTALL" \
make test-collect-report || rc=$?
rc=${rc:-0}

echo "=== BEGIN import-repair-report (head -40) ==="
head -n 40 "$IMPORT_REPAIR_REPORT" || true
echo "=== END import-repair-report ==="

exit "$rc"

