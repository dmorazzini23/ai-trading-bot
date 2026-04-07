#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${AI_TRADING_RUNTIME_DIR:-/var/lib/ai-trading-bot/runtime}"
RETENTION_DAYS="${AI_TRADING_RUNTIME_JSONL_RETENTION_DAYS:-7}"

prune_jsonl_file() {
  local path="$1"
  local max_bytes="$2"
  local keep_lines="$3"

  [[ -f "$path" ]] || return 0
  [[ "$keep_lines" -gt 0 ]] || return 0
  [[ "$max_bytes" -gt 0 ]] || return 0

  local size
  size="$(stat -c '%s' "$path" 2>/dev/null || echo 0)"
  [[ "${size:-0}" -gt "$max_bytes" ]] || return 0

  local ts backup tmp
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  backup="${path}.bak.${ts}"
  tmp="${path}.tmp.${ts}"

  tail -n "$keep_lines" "$path" > "$tmp"
  mv "$path" "$backup"
  mv "$tmp" "$path"

  chmod --reference="$backup" "$path" 2>/dev/null || true
  chown --reference="$backup" "$path" 2>/dev/null || true
  gzip -f "$backup" 2>/dev/null || true

  find "$(dirname "$path")" \
    -maxdepth 1 \
    -type f \
    -name "$(basename "$path").bak.*.gz" \
    -mtime +"$RETENTION_DAYS" \
    -delete \
    2>/dev/null || true
}

prune_jsonl_file \
  "${RUNTIME_DIR}/decision_records.jsonl" \
  "${AI_TRADING_DECISION_RECORDS_MAX_BYTES:-268435456}" \
  "${AI_TRADING_DECISION_RECORDS_KEEP_LINES:-50000}"

prune_jsonl_file \
  "${RUNTIME_DIR}/config_snapshots.jsonl" \
  "${AI_TRADING_CONFIG_SNAPSHOTS_MAX_BYTES:-268435456}" \
  "${AI_TRADING_CONFIG_SNAPSHOTS_KEEP_LINES:-30000}"

prune_jsonl_file \
  "${RUNTIME_DIR}/gate_effectiveness.jsonl" \
  "${AI_TRADING_GATE_EFFECTIVENESS_MAX_BYTES:-268435456}" \
  "${AI_TRADING_GATE_EFFECTIVENESS_KEEP_LINES:-300000}"

prune_jsonl_file \
  "${RUNTIME_DIR}/ml_shadow_predictions.jsonl" \
  "${AI_TRADING_ML_SHADOW_PREDICTIONS_MAX_BYTES:-268435456}" \
  "${AI_TRADING_ML_SHADOW_PREDICTIONS_KEEP_LINES:-100000}"
