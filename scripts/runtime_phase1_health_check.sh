#!/usr/bin/env bash
set -euo pipefail

N="${N:-5000}"
RUNTIME_DIR="${AI_TRADING_RUNTIME_DIR:-/var/lib/ai-trading-bot/runtime}"
REPORT_DIR="${REPORT_DIR:-${RUNTIME_DIR}/research_reports}"
DECISION_FILE="${DECISION_FILE:-${RUNTIME_DIR}/decision_records.jsonl}"
GATE_EFFECTIVENESS_JSONL="${GATE_EFFECTIVENESS_JSONL:-${RUNTIME_DIR}/gate_effectiveness.jsonl}"
GATE_EFFECTIVENESS_SUMMARY="${GATE_EFFECTIVENESS_SUMMARY:-${RUNTIME_DIR}/gate_effectiveness_summary.json}"
SHADOW_PREDICTIONS_FILE="${SHADOW_PREDICTIONS_FILE:-${RUNTIME_DIR}/ml_shadow_predictions.jsonl}"
ENV_RUNTIME_FILE="${ENV_RUNTIME_FILE:-/home/aiuser/ai-trading-bot/.env.runtime}"

AUTH_HALT_MAX_RATE="${AUTH_HALT_MAX_RATE:-0.35}"
OK_TRADE_MIN_RATE="${OK_TRADE_MIN_RATE:-0.01}"
RATE_ALERT_MIN_ROWS="${RATE_ALERT_MIN_ROWS:-100}"
DECISION_STALE_MAX_AGE_MINUTES="${DECISION_STALE_MAX_AGE_MINUTES:-90}"
SUPPRESS_OFFHOURS_STALE_DECISION_ALERTS="${SUPPRESS_OFFHOURS_STALE_DECISION_ALERTS:-1}"
FAIL_ON_STALE_DECISION_WINDOW_DURING_RTH="${FAIL_ON_STALE_DECISION_WINDOW_DURING_RTH:-1}"
SUPPRESS_OFFHOURS_EMPTY_DECISION_ALERTS="${SUPPRESS_OFFHOURS_EMPTY_DECISION_ALERTS:-1}"
FAIL_ON_EMPTY_DECISION_WINDOW_DURING_RTH="${FAIL_ON_EMPTY_DECISION_WINDOW_DURING_RTH:-1}"
RTH_TZ="${RTH_TZ:-America/New_York}"
RTH_START_HHMM="${RTH_START_HHMM:-0930}"
RTH_END_HHMM="${RTH_END_HHMM:-1600}"
REPORT_MAX_AGE_MINUTES="${REPORT_MAX_AGE_MINUTES:-2160}" # 36h
GATE_FILES_MAX_AGE_MINUTES="${GATE_FILES_MAX_AGE_MINUTES:-180}" # 3h
SHADOW_FILE_MAX_AGE_MINUTES="${SHADOW_FILE_MAX_AGE_MINUTES:-180}" # 3h

failures=0

require_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${name}"
    exit 2
  fi
}

fail() {
  echo "FAIL: $*"
  failures=$((failures + 1))
}

ok() {
  echo "OK: $*"
}

is_truthy() {
  local value="${1:-}"
  case "${value,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

rate_gt() {
  local lhs="$1"
  local rhs="$2"
  awk -v l="${lhs}" -v r="${rhs}" 'BEGIN { exit !(l > r) }'
}

rate_lt() {
  local lhs="$1"
  local rhs="$2"
  awk -v l="${lhs}" -v r="${rhs}" 'BEGIN { exit !(l < r) }'
}

file_age_minutes() {
  local path="$1"
  local now_s mtime_s
  now_s="$(date +%s)"
  mtime_s="$(stat -c %Y "${path}")"
  echo $(((now_s - mtime_s) / 60))
}

read_runtime_env_value() {
  local key="$1"
  local value=""
  if [[ -f "${ENV_RUNTIME_FILE}" ]]; then
    value="$(
      awk -v key="${key}" '
        /^[[:space:]]*#/ { next }
        index($0, "=") == 0 { next }
        {
          split($0, kv, "=")
          k = kv[1]
          gsub(/[[:space:]]/, "", k)
          if (k != key) {
            next
          }
          v = substr($0, index($0, "=") + 1)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
          if ((substr(v, 1, 1) == "\"" && substr(v, length(v), 1) == "\"") || (substr(v, 1, 1) == "'"'"'" && substr(v, length(v), 1) == "'"'"'")) {
            v = substr(v, 2, length(v) - 2)
          }
          print v
        }
      ' "${ENV_RUNTIME_FILE}" | tail -n 1
    )"
  fi
  echo "${value}"
}

report_age_minutes() {
  local path="$1"
  local ts_raw epoch now_s
  ts_raw="$(jq -r '.ts // empty' "${path}" 2>/dev/null || true)"
  if [[ -n "${ts_raw}" ]]; then
    epoch="$(date -u -d "${ts_raw}" +%s 2>/dev/null || true)"
  else
    epoch=""
  fi
  if [[ -z "${epoch}" ]]; then
    epoch="$(stat -c %Y "${path}")"
  fi
  now_s="$(date +%s)"
  echo $(((now_s - epoch) / 60))
}

timestamp_age_minutes() {
  local ts_raw="$1"
  local epoch now_s
  epoch="$(date -u -d "${ts_raw}" +%s 2>/dev/null || true)"
  if [[ -z "${epoch}" ]]; then
    return 1
  fi
  now_s="$(date +%s)"
  echo $(((now_s - epoch) / 60))
}

is_regular_trading_hours_now() {
  local dow hhmm
  dow="$(TZ="${RTH_TZ}" date +%u)"
  hhmm="$(TZ="${RTH_TZ}" date +%H%M)"
  if ((10#${dow} < 1 || 10#${dow} > 5)); then
    return 1
  fi
  if ((10#${hhmm} < 10#${RTH_START_HHMM} || 10#${hhmm} > 10#${RTH_END_HHMM})); then
    return 1
  fi
  return 0
}

require_cmd jq
require_cmd awk
require_cmd tail
require_cmd sort
require_cmd uniq
require_cmd head
require_cmd stat
require_cmd find
require_cmd date

echo "== runtime phase-1 health check =="
echo "runtime_dir=${RUNTIME_DIR}"
echo "decision_file=${DECISION_FILE}"
echo "report_dir=${REPORT_DIR}"
echo "window_rows=${N}"
echo "thresholds: auth_halt_max_rate=${AUTH_HALT_MAX_RATE}, ok_trade_min_rate=${OK_TRADE_MIN_RATE}"
echo "decision_window: min_rows_for_rate_alerts=${RATE_ALERT_MIN_ROWS}, stale_max_age_minutes=${DECISION_STALE_MAX_AGE_MINUTES}"

skip_decision_derived_artifact_checks=0

if [[ ! -s "${DECISION_FILE}" ]]; then
  if is_regular_trading_hours_now; then
    if is_truthy "${FAIL_ON_EMPTY_DECISION_WINDOW_DURING_RTH}"; then
      fail "decision records missing or empty during regular trading hours: ${DECISION_FILE}"
    else
      echo "WARN: decision records missing or empty during regular trading hours: ${DECISION_FILE}"
    fi
  else
    if is_truthy "${SUPPRESS_OFFHOURS_EMPTY_DECISION_ALERTS}"; then
      echo "INFO: decision records missing or empty off-hours; decision-rate checks suppressed"
    else
      fail "decision records missing or empty: ${DECISION_FILE}"
    fi
  fi
  skip_decision_derived_artifact_checks=1
else
  latest_decision_ts="$(
    tail -n "${N}" "${DECISION_FILE}" \
      | jq -s -r 'map(.bar_ts // .ts // .timestamp // empty) | if length > 0 then .[-1] else empty end'
  )"
  decision_age_m=""
  if [[ -n "${latest_decision_ts}" ]]; then
    decision_age_m="$(timestamp_age_minutes "${latest_decision_ts}" || true)"
  fi
  if [[ -z "${decision_age_m}" ]]; then
    decision_age_m="$(file_age_minutes "${DECISION_FILE}")"
  fi

  stats_json="$(
    tail -n "${N}" "${DECISION_FILE}" \
      | jq -s '{
          rows: length,
          ok_trade_rows: (map(select((.gates // []) | index("OK_TRADE"))) | length),
          auth_halt_rows: (map(select((.gates // []) | index("AUTH_HALT"))) | length),
          order_pacing_cap_rows: (map(select((.gates // []) | index("ORDER_PACING_CAP_BLOCK"))) | length)
        }'
  )"
  rows="$(jq -r '.rows' <<<"${stats_json}")"
  ok_rows="$(jq -r '.ok_trade_rows' <<<"${stats_json}")"
  auth_rows="$(jq -r '.auth_halt_rows' <<<"${stats_json}")"
  pacing_rows="$(jq -r '.order_pacing_cap_rows' <<<"${stats_json}")"

  if [[ "${rows}" -le 0 ]]; then
    fail "decision window has zero rows"
  else
    skip_rate_alerts=0

    if [[ "${rows}" -lt "${RATE_ALERT_MIN_ROWS}" ]]; then
      echo "INFO: rate alerts skipped (rows=${rows} < min=${RATE_ALERT_MIN_ROWS})"
      skip_rate_alerts=1
    fi

    if [[ "${decision_age_m}" -gt "${DECISION_STALE_MAX_AGE_MINUTES}" ]]; then
      if is_regular_trading_hours_now; then
        if is_truthy "${FAIL_ON_STALE_DECISION_WINDOW_DURING_RTH}"; then
          fail "decision window stale during regular trading hours (${decision_age_m}m > ${DECISION_STALE_MAX_AGE_MINUTES}m)"
        else
          echo "WARN: decision window stale during regular trading hours (${decision_age_m}m)"
        fi
      else
        if is_truthy "${SUPPRESS_OFFHOURS_STALE_DECISION_ALERTS}"; then
          echo "INFO: decision window stale off-hours (${decision_age_m}m); rate alerts suppressed"
        else
          fail "decision window stale off-hours (${decision_age_m}m > ${DECISION_STALE_MAX_AGE_MINUTES}m)"
        fi
      fi
      skip_rate_alerts=1
    fi

    auth_rate="$(awk -v n="${auth_rows}" -v d="${rows}" 'BEGIN { printf "%.6f", (d > 0 ? n / d : 0.0) }')"
    ok_rate="$(awk -v n="${ok_rows}" -v d="${rows}" 'BEGIN { printf "%.6f", (d > 0 ? n / d : 0.0) }')"
    auth_pct="$(awk -v r="${auth_rate}" 'BEGIN { printf "%.2f", r * 100.0 }')"
    ok_pct="$(awk -v r="${ok_rate}" 'BEGIN { printf "%.2f", r * 100.0 }')"
    echo "window_stats: rows=${rows} auth_halt_rows=${auth_rows} ok_trade_rows=${ok_rows} order_pacing_cap_rows=${pacing_rows} decision_age_minutes=${decision_age_m}"
    echo "window_rates: auth_halt_rate=${auth_rate} (${auth_pct}%), ok_trade_rate=${ok_rate} (${ok_pct}%)"

    if [[ "${skip_rate_alerts}" -eq 1 ]]; then
      echo "INFO: AUTH_HALT/OK_TRADE threshold checks skipped for this decision window"
    else
      if rate_gt "${auth_rate}" "${AUTH_HALT_MAX_RATE}"; then
        fail "AUTH_HALT spike detected: ${auth_rate} > ${AUTH_HALT_MAX_RATE}"
      else
        ok "AUTH_HALT rate within threshold"
      fi

      if rate_lt "${ok_rate}" "${OK_TRADE_MIN_RATE}"; then
        fail "OK_TRADE collapse detected: ${ok_rate} < ${OK_TRADE_MIN_RATE}"
      else
        ok "OK_TRADE rate within threshold"
      fi
    fi

    echo "top_gates:"
    tail -n "${N}" "${DECISION_FILE}" | jq -r '.gates[]?' | sort | uniq -c | sort -nr | head -n 15
  fi
fi

if [[ "${skip_decision_derived_artifact_checks}" -eq 1 ]]; then
  echo "INFO: gate-effectiveness artifact checks skipped (no recent decision window)"
else
  for path in "${GATE_EFFECTIVENESS_JSONL}" "${GATE_EFFECTIVENESS_SUMMARY}"; do
    if [[ ! -s "${path}" ]]; then
      fail "missing gate effectiveness artifact: ${path}"
      continue
    fi
    age_m="$(file_age_minutes "${path}")"
    if [[ "${age_m}" -gt "${GATE_FILES_MAX_AGE_MINUTES}" ]]; then
      fail "stale gate effectiveness artifact (${age_m}m): ${path}"
    else
      ok "gate effectiveness artifact present and fresh (${age_m}m): ${path}"
    fi
  done
fi

latest_report="$(
  find "${REPORT_DIR}" -maxdepth 1 -type f -name 'after_hours_training_*.json' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
)"
if [[ -z "${latest_report}" ]]; then
  fail "no after-hours training report found in ${REPORT_DIR}"
else
  report_age_m="$(report_age_minutes "${latest_report}")"
  if [[ "${report_age_m}" -gt "${REPORT_MAX_AGE_MINUTES}" ]]; then
    fail "after-hours report stale (${report_age_m}m): ${latest_report}"
  else
    ok "after-hours report fresh (${report_age_m}m): ${latest_report}"
  fi
fi

shadow_enabled="${AI_TRADING_ML_SHADOW_ENABLED:-}"
if [[ -z "${shadow_enabled}" ]]; then
  shadow_enabled="$(read_runtime_env_value "AI_TRADING_ML_SHADOW_ENABLED")"
fi
shadow_enabled="${shadow_enabled:-0}"

if is_truthy "${shadow_enabled}"; then
  if [[ "${skip_decision_derived_artifact_checks}" -eq 1 ]]; then
    echo "INFO: shadow artifact check skipped (no recent decision window)"
  else
    if [[ ! -s "${SHADOW_PREDICTIONS_FILE}" ]]; then
      fail "shadow enabled but predictions artifact missing: ${SHADOW_PREDICTIONS_FILE}"
    else
      shadow_age_m="$(file_age_minutes "${SHADOW_PREDICTIONS_FILE}")"
      if [[ "${shadow_age_m}" -gt "${SHADOW_FILE_MAX_AGE_MINUTES}" ]]; then
        fail "shadow predictions stale (${shadow_age_m}m): ${SHADOW_PREDICTIONS_FILE}"
      else
        ok "shadow predictions present and fresh (${shadow_age_m}m): ${SHADOW_PREDICTIONS_FILE}"
      fi
    fi
  fi
else
  echo "INFO: shadow check skipped (AI_TRADING_ML_SHADOW_ENABLED=${shadow_enabled})"
fi

if [[ "${failures}" -gt 0 ]]; then
  echo "RESULT: FAIL (${failures} checks failed)"
  exit 1
fi

echo "RESULT: PASS"
exit 0
