#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m ai_trading --interval 30 &
BOT_PID=$!

sleep 5
kill -TERM "$BOT_PID" 2>/dev/null || true

wait "$BOT_PID"
echo "Process ${BOT_PID} exited cleanly after SIGTERM"
