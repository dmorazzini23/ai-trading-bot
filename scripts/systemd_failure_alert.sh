#!/usr/bin/env bash
set -euo pipefail

if [[ "${AI_TRADING_SYSTEMD_FAILURE_ALERTS_ENABLED:-1}" != "1" ]]; then
  exit 0
fi

FAILED_UNIT="${1:-unknown-unit}"
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOSTNAME_VALUE="$(hostname -f 2>/dev/null || hostname)"
RUNTIME_ROOT="${AI_TRADING_RUNTIME_DIR:-/var/lib/ai-trading-bot/runtime}"
ALERT_EVENTS_PATH="${AI_TRADING_SYSTEMD_FAILURE_ALERT_RUNTIME_PATH:-${RUNTIME_ROOT}/systemd_failure_alerts.jsonl}"
WEBHOOK_URL="${AI_TRADING_SLACK_WEBHOOK_URL:-${SLACK_WEBHOOK_URL:-}}"

mkdir -p "$(dirname "${ALERT_EVENTS_PATH}")"

json_line="$(python3 - "${NOW_UTC}" "${FAILED_UNIT}" "${HOSTNAME_VALUE}" <<'PY'
import json
import sys

ts, unit_name, host_name = sys.argv[1], sys.argv[2], sys.argv[3]
print(
    json.dumps(
        {
            "ts": ts,
            "event": "systemd_unit_failure",
            "unit": unit_name,
            "host": host_name,
        },
        separators=(",", ":"),
    )
)
PY
)"

printf '%s\n' "${json_line}" >> "${ALERT_EVENTS_PATH}" || true

if [[ -z "${WEBHOOK_URL}" ]]; then
  logger -t ai-trading -p user.warning "SYSTEMD_FAILURE_ALERT missing webhook; unit=${FAILED_UNIT}"
  exit 0
fi

slack_payload="$(python3 - "${NOW_UTC}" "${FAILED_UNIT}" "${HOSTNAME_VALUE}" <<'PY'
import json
import sys

ts, unit_name, host_name = sys.argv[1], sys.argv[2], sys.argv[3]
message = (
    f":rotating_light: ai-trading systemd failure\n"
    f"- unit: `{unit_name}`\n"
    f"- host: `{host_name}`\n"
    f"- ts: `{ts}`"
)
print(json.dumps({"text": message}, separators=(",", ":")))
PY
)"

if ! curl -fsS -m 10 \
  -H "Content-Type: application/json" \
  --data "${slack_payload}" \
  "${WEBHOOK_URL}" >/dev/null; then
  logger -t ai-trading -p user.err "SYSTEMD_FAILURE_ALERT webhook post failed; unit=${FAILED_UNIT}"
fi

