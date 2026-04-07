#!/usr/bin/env bash
set -euo pipefail

if [[ "${AI_TRADING_BACKUP_S3_SYNC_ENABLED:-1}" != "1" ]]; then
  exit 0
fi

RUNTIME_DIR="${AI_TRADING_RUNTIME_DIR:-/var/lib/ai-trading-bot/runtime}"
S3_BUCKET="${AI_TRADING_BACKUP_S3_BUCKET:-}"
S3_PREFIX="${AI_TRADING_BACKUP_S3_PREFIX:-pruned/}"
AWS_REGION="${AI_TRADING_BACKUP_S3_REGION:-${AWS_REGION:-${AI_TRADING_AWS_REGION:-us-east-2}}}"

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
