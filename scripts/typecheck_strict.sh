#!/usr/bin/env bash
set -euo pipefail

# High-signal strict type checks for critical runtime paths.
python3 -m mypy --config-file mypy_strict.ini \
  ai_trading/config/management.py \
  ai_trading/health_payload.py \
  ai_trading/health.py \
  ai_trading/logging/emit_once.py \
  ai_trading/strategy_allocator.py \
  ai_trading/rl/module.py

# Runtime-critical modules that are not yet strict-clean still get explicit
# baseline type coverage in this gate.
python3 -m mypy --config-file mypy.ini \
  --follow-imports=skip \
  --disable-error-code call-overload \
  --disable-error-code name-defined \
  --disable-error-code has-type \
  ai_trading/__main__.py \
  ai_trading/app.py \
  ai_trading/core/bot_engine.py \
  ai_trading/data/provider_monitor.py \
  ai_trading/main.py \
  ai_trading/logging/__init__.py \
  ai_trading/http/pooling.py \
  ai_trading/data/fetch/__init__.py
