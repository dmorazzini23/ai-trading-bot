#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SRC="${REPO_ROOT}/packaging/systemd/ai-trading-metrics-forwarder.service"
SERVICE_DST="/etc/systemd/system/ai-trading-metrics-forwarder.service"
PROM_CFG_DST="/etc/ai-trading/prometheus-forwarder.yml"
ENV_FILE="${REPO_ROOT}/.env.runtime"

if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

LOCAL_PORT="${AI_TRADING_PROMETHEUS_LOCAL_PORT:-19090}"
API_PORT="${API_PORT:-9001}"
SCRAPE_HOST="${AI_TRADING_PROMETHEUS_SCRAPE_HOST:-127.0.0.1}"
SCRAPE_INTERVAL="${AI_TRADING_PROMETHEUS_SCRAPE_INTERVAL:-30s}"
METRIC_REGEX="${AI_TRADING_PROMETHEUS_METRIC_REGEX:-ai_trading_.*|bot_.*|orders_.*|alpaca_.*}"
RW_QUEUE_CAPACITY="${AI_TRADING_PROM_RW_QUEUE_CAPACITY:-25000}"
RW_QUEUE_MAX_SHARDS="${AI_TRADING_PROM_RW_QUEUE_MAX_SHARDS:-4}"
RW_QUEUE_MIN_SHARDS="${AI_TRADING_PROM_RW_QUEUE_MIN_SHARDS:-1}"
RW_QUEUE_MAX_SAMPLES_PER_SEND="${AI_TRADING_PROM_RW_QUEUE_MAX_SAMPLES_PER_SEND:-5000}"
RW_QUEUE_BATCH_SEND_DEADLINE="${AI_TRADING_PROM_RW_QUEUE_BATCH_SEND_DEADLINE:-20s}"
RW_SEND_METADATA_RAW="${AI_TRADING_PROM_RW_SEND_METADATA:-0}"

case "${RW_SEND_METADATA_RAW,,}" in
  1|true|yes|on)
    RW_SEND_METADATA="true"
    ;;
  *)
    RW_SEND_METADATA="false"
    ;;
esac

REMOTE_WRITE_URL="${AI_TRADING_PROM_REMOTE_WRITE_URL:-${PROM_REMOTE_WRITE_URL:-}}"
REMOTE_WRITE_USERNAME="${AI_TRADING_PROM_REMOTE_WRITE_USERNAME:-${PROM_REMOTE_WRITE_USERNAME:-}}"
REMOTE_WRITE_PASSWORD="${AI_TRADING_PROM_REMOTE_WRITE_PASSWORD:-${PROM_REMOTE_WRITE_PASSWORD:-}}"

_mask() {
  local value="${1:-}"
  if [[ -z "${value}" ]]; then
    echo "<empty>"
    return
  fi
  local prefix="${value:0:8}"
  local suffix="${value: -4}"
  echo "${prefix}...${suffix}"
}

_discover_from_grafana() {
  if [[ -n "${REMOTE_WRITE_URL}" && -n "${REMOTE_WRITE_USERNAME}" ]]; then
    return
  fi
  if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
    return
  fi
  local grafana_url="${AI_TRADING_GRAFANA_URL:-}"
  local grafana_uid="${AI_TRADING_GRAFANA_PROMETHEUS_UID:-}"
  local grafana_token="${AI_TRADING_GRAFANA_API_TOKEN:-}"
  if [[ -z "${grafana_url}" || -z "${grafana_uid}" || -z "${grafana_token}" ]]; then
    return
  fi

  local endpoint="${grafana_url%/}/api/datasources/uid/${grafana_uid}"
  local payload
  payload="$(curl -fsS -H "Authorization: Bearer ${grafana_token}" "${endpoint}" 2>/dev/null || true)"
  if [[ -z "${payload}" ]]; then
    return
  fi

  local ds_url ds_user
  ds_url="$(printf "%s" "${payload}" | jq -r '.datasource.url // .url // empty')"
  ds_user="$(printf "%s" "${payload}" | jq -r '.datasource.basicAuthUser // .basicAuthUser // empty')"

  if [[ -z "${REMOTE_WRITE_URL}" && -n "${ds_url}" ]]; then
    if [[ "${ds_url}" == */api/prom ]]; then
      REMOTE_WRITE_URL="${ds_url%/api/prom}/api/prom/push"
    else
      REMOTE_WRITE_URL="${ds_url}"
    fi
  fi
  if [[ -z "${REMOTE_WRITE_USERNAME}" && -n "${ds_user}" ]]; then
    REMOTE_WRITE_USERNAME="${ds_user}"
  fi
}

_discover_from_grafana

if [[ -z "${REMOTE_WRITE_URL}" || -z "${REMOTE_WRITE_USERNAME}" || -z "${REMOTE_WRITE_PASSWORD}" ]]; then
  echo "Missing remote_write credentials."
  echo "Required:"
  echo "  AI_TRADING_PROM_REMOTE_WRITE_URL"
  echo "  AI_TRADING_PROM_REMOTE_WRITE_USERNAME"
  echo "  AI_TRADING_PROM_REMOTE_WRITE_PASSWORD"
  echo
  echo "Note: AI_TRADING_GRAFANA_API_TOKEN is only used for datasource discovery."
  echo "It is usually NOT valid as remote_write password."
  echo
  echo "Current:"
  echo "  remote_write_url=${REMOTE_WRITE_URL:-<empty>}"
  echo "  remote_write_username=${REMOTE_WRITE_USERNAME:-<empty>}"
  echo "  remote_write_password=$(_mask "${REMOTE_WRITE_PASSWORD}")"
  exit 1
fi

if ! command -v prometheus >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y prometheus
fi

TMP_CFG="$(mktemp)"
trap 'rm -f "${TMP_CFG}"' EXIT

cat >"${TMP_CFG}" <<EOF
global:
  scrape_interval: "${SCRAPE_INTERVAL}"
  external_labels:
    service: "ai-trading"

scrape_configs:
  - job_name: "ai_trading_runtime"
    metrics_path: "/metrics"
    static_configs:
      - targets:
          - "${SCRAPE_HOST}:${API_PORT}"
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: "${METRIC_REGEX}"
        action: keep

remote_write:
  - url: "${REMOTE_WRITE_URL}"
    basic_auth:
      username: "${REMOTE_WRITE_USERNAME}"
      password: "${REMOTE_WRITE_PASSWORD}"
    metadata_config:
      send: ${RW_SEND_METADATA}
    queue_config:
      capacity: ${RW_QUEUE_CAPACITY}
      max_shards: ${RW_QUEUE_MAX_SHARDS}
      min_shards: ${RW_QUEUE_MIN_SHARDS}
      max_samples_per_send: ${RW_QUEUE_MAX_SAMPLES_PER_SEND}
      batch_send_deadline: ${RW_QUEUE_BATCH_SEND_DEADLINE}
EOF

sudo install -d -m 0755 /etc/ai-trading
sudo install -d -m 0700 -o aiuser -g aiuser /var/lib/ai-trading-bot/prometheus
sudo install -m 0640 -o root -g aiuser "${TMP_CFG}" "${PROM_CFG_DST}"
sudo install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-metrics-forwarder.service

if [[ -f "${REPO_ROOT}/.env" ]]; then
  if grep -q "^AI_TRADING_PROMETHEUS_URL=" "${REPO_ROOT}/.env"; then
    sed -i "s|^AI_TRADING_PROMETHEUS_URL=.*|AI_TRADING_PROMETHEUS_URL=http://127.0.0.1:${LOCAL_PORT}|" "${REPO_ROOT}/.env"
  else
    echo "AI_TRADING_PROMETHEUS_URL=http://127.0.0.1:${LOCAL_PORT}" >> "${REPO_ROOT}/.env"
  fi
fi

if [[ -f "${REPO_ROOT}/.env.runtime" ]]; then
  if grep -q "^AI_TRADING_PROMETHEUS_URL=" "${REPO_ROOT}/.env.runtime"; then
    sed -i "s|^AI_TRADING_PROMETHEUS_URL=.*|AI_TRADING_PROMETHEUS_URL=http://127.0.0.1:${LOCAL_PORT}|" "${REPO_ROOT}/.env.runtime"
  else
    echo "AI_TRADING_PROMETHEUS_URL=http://127.0.0.1:${LOCAL_PORT}" >> "${REPO_ROOT}/.env.runtime"
  fi
fi

echo "=== metrics forwarder status ==="
sudo systemctl status ai-trading-metrics-forwarder.service --no-pager -l | sed -n '1,120p'
echo "=== local prometheus ready ==="
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/-/ready" || true
echo
echo "=== sample query (local prometheus) ==="
curl -fsS "http://127.0.0.1:${LOCAL_PORT}/api/v1/query?query=ai_trading_slippage_drag_bps" | jq .
