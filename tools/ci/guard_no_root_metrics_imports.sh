#!/usr/bin/env bash
set -euo pipefail
if git grep -nE '^(from|import)\s+metrics_logger\b' -- 'ai_trading/**/*.py' | grep -q .; then
  echo "ERROR: root-level metrics_logger import detected; use ai_trading.telemetry paths."
  git grep -nE '^(from|import)\s+metrics_logger\b' -- 'ai_trading/**/*.py'
  exit 1
fi