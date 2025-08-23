#!/usr/bin/env bash
# AI-AGENT-REF: CI smoke script
# Run test-collect-report, print the first 40 lines of the artifact,
# and exit 0/101 based on FAIL_ON_IMPORT_ERRORS.
set -uo pipefail

REPORT="${IMPORT_REPAIR_REPORT:-artifacts/import-repair-report.md}"
TOP_N="${TOP_N:-5}"

echo "==> CI smoke: make test-collect-report (TOP_N=${TOP_N}, FAIL_ON_IMPORT_ERRORS=${FAIL_ON_IMPORT_ERRORS:-}, DISABLE_ENV_ASSERT=${DISABLE_ENV_ASSERT:-})"
status=0
make test-collect-report || status=$?

echo "==> Artifact preview (first 40 lines): ${REPORT}"
if [[ -f "${REPORT}" ]]; then
  sed -n '1,40p' "${REPORT}"
else
  echo "(artifact not found at ${REPORT})"
fi

# Normalize exit behaviour.
if [[ -n "${FAIL_ON_IMPORT_ERRORS:-}" ]]; then
  if [[ ${status} -eq 101 ]]; then
    echo "==> Import errors found and FAIL_ON_IMPORT_ERRORS=1; exiting 101."
    exit 101
  elif [[ ${status} -eq 0 ]]; then
    echo "==> No import errors detected; exiting 0."
    exit 0
  else
    echo "==> Collector returned unexpected status ${status}; bubbling it up."
    exit ${status}
  fi
else
  echo "==> FAIL_ON_IMPORT_ERRORS not set; exiting 0 (collector status=${status})."
  exit 0
fi
