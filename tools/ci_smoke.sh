#!/usr/bin/env bash
set -euo pipefail
# AI-AGENT-REF: deterministic import smoke script

: "${TOP_N:=5}"
: "${FAIL_ON_IMPORT_ERRORS:=0}"
: "${DISABLE_ENV_ASSERT:=0}"
: "${IMPORT_REPAIR_REPORT:=artifacts/import-repair-report.md}"
: "${SKIP_INSTALL:=0}"

mkdir -p "$(dirname "$IMPORT_REPAIR_REPORT")"

# optionally skip install for fast signal
if [[ "${SKIP_INSTALL}" != "1" ]]; then
  make ensure-runtime
fi

# run collection + harvest; never die before we can print the header
rc=0
DISABLE_ENV_ASSERT="${DISABLE_ENV_ASSERT}" \
TOP_N="${TOP_N}" \
FAIL_ON_IMPORT_ERRORS="${FAIL_ON_IMPORT_ERRORS}" \
make test-collect-report || rc=$?

echo "=== BEGIN import-repair-report (head -40) ==="
if [[ -f "$IMPORT_REPAIR_REPORT" ]]; then
  head -n 40 "$IMPORT_REPAIR_REPORT" || true
else
  echo "report missing: $IMPORT_REPAIR_REPORT"
fi
echo "=== END import-repair-report ==="

exit "${rc}"
