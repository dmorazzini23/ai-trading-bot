#!/usr/bin/env bash
set -euo pipefail

if [[ "${AI_TRADING_BACKUP_S3_SYNC_ENABLED:-1}" != "1" ]]; then
  exit 0
fi

RUNTIME_DIR="${AI_TRADING_RUNTIME_DIR:-/var/lib/ai-trading-bot/runtime}"
S3_BUCKET="${AI_TRADING_BACKUP_S3_BUCKET:-}"
S3_PREFIX="${AI_TRADING_BACKUP_S3_PREFIX:-pruned/}"
AWS_REGION="${AI_TRADING_BACKUP_S3_REGION:-${AWS_REGION:-${AI_TRADING_AWS_REGION:-us-east-2}}}"
RETENTION_ENABLED="${AI_TRADING_BACKUP_S3_RETENTION_ENABLED:-1}"
RETENTION_DAYS="${AI_TRADING_BACKUP_S3_RETENTION_DAYS:-30}"
RETENTION_MAX_DELETES="${AI_TRADING_BACKUP_S3_RETENTION_MAX_DELETES:-500}"

if [[ -z "$S3_BUCKET" ]]; then
  echo "AI_TRADING_BACKUP_S3_BUCKET is required" >&2
  exit 2
fi

# Normalize prefix so object keys are stable.
S3_PREFIX="${S3_PREFIX#/}"
if [[ -n "$S3_PREFIX" && "$S3_PREFIX" != */ ]]; then
  S3_PREFIX="${S3_PREFIX}/"
fi

DESTINATION="s3://${S3_BUCKET}/${S3_PREFIX}"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

staged=0
while IFS= read -r -d '' file_path; do
  rel_path="${file_path#${RUNTIME_DIR}/}"
  target_dir="${tmpdir}/$(dirname "${rel_path}")"
  mkdir -p "${target_dir}"
  # Prefer hard-links to avoid copy overhead; fall back to a regular copy if needed.
  ln "${file_path}" "${tmpdir}/${rel_path}" 2>/dev/null \
    || cp -p "${file_path}" "${tmpdir}/${rel_path}"
  staged=1
done < <(find "${RUNTIME_DIR}" -type f -name "*.bak.*.gz" -readable -print0 2>/dev/null)

if [[ "${staged}" -eq 0 ]]; then
  exit 0
fi

aws s3 sync "${tmpdir}" "${DESTINATION}" \
  --region "${AWS_REGION}" \
  --sse AES256 \
  --no-progress \
  --only-show-errors

if [[ "${RETENTION_ENABLED}" != "1" ]]; then
  exit 0
fi

if ! [[ "${RETENTION_DAYS}" =~ ^[0-9]+$ ]] || ! [[ "${RETENTION_MAX_DELETES}" =~ ^[0-9]+$ ]]; then
  echo "S3 retention settings are invalid; skipping retention pass" >&2
  exit 0
fi

if [[ "${RETENTION_DAYS}" -le 0 ]]; then
  exit 0
fi

cutoff_epoch="$(date -u -d "-${RETENTION_DAYS} days" +%s)"
deleted=0

while read -r date_str time_str _size key; do
  [[ -n "${key:-}" ]] || continue
  [[ "${key}" == *.bak.*.gz ]] || continue

  object_epoch="$(date -u -d "${date_str} ${time_str} UTC" +%s 2>/dev/null || true)"
  [[ -n "${object_epoch}" ]] || continue
  if [[ "${object_epoch}" -ge "${cutoff_epoch}" ]]; then
    continue
  fi

  if [[ "${deleted}" -ge "${RETENTION_MAX_DELETES}" ]]; then
    echo "S3 retention delete cap reached (${RETENTION_MAX_DELETES}); stopping pass"
    break
  fi

  if aws s3 rm "s3://${S3_BUCKET}/${key}" --region "${AWS_REGION}" --only-show-errors >/dev/null 2>&1; then
    deleted=$((deleted + 1))
  fi
done < <(aws s3 ls "${DESTINATION}" --recursive --region "${AWS_REGION}" 2>/dev/null || true)

if [[ "${deleted}" -gt 0 ]]; then
  echo "S3 retention removed ${deleted} backup object(s) older than ${RETENTION_DAYS} day(s)"
fi
