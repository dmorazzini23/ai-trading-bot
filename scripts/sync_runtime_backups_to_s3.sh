#!/usr/bin/env bash
set -euo pipefail

if [[ "${AI_TRADING_BACKUP_S3_SYNC_ENABLED:-1}" != "1" ]]; then
  exit 0
fi

RUNTIME_DIR="${AI_TRADING_RUNTIME_DIR:-/var/lib/ai-trading-bot/runtime}"
S3_BUCKET="${AI_TRADING_BACKUP_S3_BUCKET:-}"
S3_PREFIX="${AI_TRADING_BACKUP_S3_PREFIX:-pruned/}"
S3_OWNER="${AI_TRADING_BACKUP_S3_EXPECTED_BUCKET_OWNER:-}"
AWS_REGION="${AI_TRADING_BACKUP_S3_REGION:-${AWS_REGION:-${AI_TRADING_AWS_REGION:-us-east-2}}}"
PYTHON_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "$S3_BUCKET" || ! "$S3_OWNER" =~ ^[0-9]{12}$ ]]; then
  echo "Backup S3 bucket and 12-digit expected bucket owner are required" >&2
  exit 2
fi

if [[ ! "$S3_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9/_-]*/?$ || "$S3_PREFIX" == *//* ]]; then
  echo "Backup S3 prefix is invalid" >&2
  exit 2
fi
if [[ -n "$S3_PREFIX" && "$S3_PREFIX" != */ ]]; then
  S3_PREFIX="${S3_PREFIX}/"
fi

backup_dir="${RUNTIME_DIR}/recovery_backups"
shopt -s nullglob
bundle=""
for candidate in "${backup_dir}"/recovery.bak.*.gz; do
  name="${candidate##*/}"
  [[ "$name" =~ ^recovery\.bak\.[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}\.gz$ ]] || continue
  [[ -f "$candidate" && ! -L "$candidate" ]] || continue
  if [[ -z "$bundle" || "$candidate" -nt "$bundle" ]]; then
    bundle="$candidate"
  fi
done
if [[ -z "$bundle" ]]; then
  echo "Runtime recovery bundle missing from S3 sync input" >&2
  exit 3
fi

"$PYTHON_BIN" -m ai_trading.tools.runtime_recovery_backup --verify "$bundle" >/dev/null

key="${S3_PREFIX}recovery_backups/${bundle##*/}"
readback="$(mktemp)"
trap 'rm -f "$readback"' EXIT

aws s3api put-object \
  --bucket "$S3_BUCKET" \
  --key "$key" \
  --body "$bundle" \
  --expected-bucket-owner "$S3_OWNER" \
  --server-side-encryption AES256 \
  --checksum-algorithm SHA256 \
  --region "$AWS_REGION" >/dev/null

aws s3api get-object \
  --bucket "$S3_BUCKET" \
  --key "$key" \
  --expected-bucket-owner "$S3_OWNER" \
  --region "$AWS_REGION" \
  "$readback" >/dev/null

if ! cmp -s "$bundle" "$readback"; then
  echo "S3 recovery bundle read-back mismatch" >&2
  exit 4
fi

echo "S3 recovery bundle uploaded and read back: $key"
