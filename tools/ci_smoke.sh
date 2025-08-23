#!/usr/bin/env bash
set -euo pipefail

: "${TOP_N:=5}"
: "${FAIL_ON_IMPORT_ERRORS:=0}"
: "${DISABLE_ENV_ASSERT:=}"
: "${IMPORT_REPAIR_REPORT:=artifacts/import-repair-report.md}"

# Run collect + harvest (never hard-fail the make; we control the exit below)
DISABLE_ENV_ASSERT="${DISABLE_ENV_ASSERT}" \
  make test-collect-report TOP_N="${TOP_N}" \
       FAIL_ON_IMPORT_ERRORS="${FAIL_ON_IMPORT_ERRORS}" \
       IMPORT_REPAIR_REPORT="${IMPORT_REPAIR_REPORT}" || true

echo "---- Import-Repair Report (head -n 40) ----"
if [[ -f "${IMPORT_REPAIR_REPORT}" ]]; then
  head -n 40 "${IMPORT_REPAIR_REPORT}"
else
  echo "Report not found at ${IMPORT_REPAIR_REPORT}"
fi
echo "-------------------------------------------"

# Exit decision mirrors harvester's --fail-on-errors behavior
if [[ "${FAIL_ON_IMPORT_ERRORS}" == "1" ]]; then
  # If the report has a 'Remaining import errors' section with a nonzero count, exit 101
  if grep -qE '^- Remaining import errors \(unique\): [1-9][0-9]*' "${IMPORT_REPAIR_REPORT}" 2>/dev/null; then
    exit 101
  fi
fi
exit 0
