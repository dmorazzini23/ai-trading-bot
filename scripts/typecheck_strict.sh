#!/usr/bin/env bash
set -euo pipefail

# High-signal strict type checks for critical runtime paths.
python3 -m mypy --config-file mypy_strict.ini \
  ai_trading/config/management.py \
  ai_trading/health_payload.py \
  ai_trading/health.py
