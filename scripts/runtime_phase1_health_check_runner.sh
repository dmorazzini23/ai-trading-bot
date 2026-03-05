#!/usr/bin/env bash
set -euo pipefail

CHECK_SCRIPT="${AI_TRADING_RUNTIME_HEALTHCHECK_SCRIPT:-/home/aiuser/ai-trading-bot/scripts/runtime_phase1_health_check.sh}"
LOG_TAG="${AI_TRADING_RUNTIME_HEALTHCHECK_LOG_TAG:-ai-trading-healthcheck}"
ALERT_WEBHOOK="${AI_TRADING_RUNTIME_HEALTHCHECK_ALERT_WEBHOOK:-}"
ALERT_TIMEOUT_SEC="${AI_TRADING_RUNTIME_HEALTHCHECK_ALERT_TIMEOUT_SEC:-10}"

if [[ ! -x "${CHECK_SCRIPT}" ]]; then
  logger -p user.err -t "${LOG_TAG}" "RUNTIME_HEALTHCHECK_SCRIPT_MISSING script=${CHECK_SCRIPT}"
  echo "RUNTIME_HEALTHCHECK_SCRIPT_MISSING script=${CHECK_SCRIPT}" >&2
  exit 2
fi

if output="$("${CHECK_SCRIPT}" 2>&1)"; then
  printf '%s\n' "${output}"
  logger -t "${LOG_TAG}" "RUNTIME_HEALTHCHECK_PASS"
  exit 0
fi

rc=$?
printf '%s\n' "${output}" >&2
logger -p user.err -t "${LOG_TAG}" "RUNTIME_HEALTHCHECK_FAIL rc=${rc}"

if [[ -n "${ALERT_WEBHOOK}" ]]; then
  payload=$(
    cat <<EOF
runtime_healthcheck=fail
host=$(hostname -f 2>/dev/null || hostname)
service=ai-trading-healthcheck.service
exit_code=${rc}
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

${output}
EOF
  )
  curl --silent --show-error --max-time "${ALERT_TIMEOUT_SEC}" \
    -X POST \
    -H "Content-Type: text/plain" \
    --data-binary "${payload}" \
    "${ALERT_WEBHOOK}" >/dev/null || true
fi

exit "${rc}"
