#!/usr/bin/env bash
set -euo pipefail

: "${TOP_N:=5}"
: "${FAIL_ON_IMPORT_ERRORS:=0}"
: "${DISABLE_ENV_ASSERT:=0}"
: "${IMPORT_REPAIR_REPORT:=artifacts/import-repair-report.md}"

mkdir -p "$(dirname "$IMPORT_REPAIR_REPORT")"

DISABLE_ENV_ASSERT="${DISABLE_ENV_ASSERT}" \
TOP_N="${TOP_N}" \
FAIL_ON_IMPORT_ERRORS="${FAIL_ON_IMPORT_ERRORS}" \
make test-collect-report || rc=$?
rc=${rc:-0}

echo "=== BEGIN import-repair-report (head -40) ==="
head -n 40 "$IMPORT_REPAIR_REPORT" || true
echo "=== END import-repair-report ==="

exit "$rc"
