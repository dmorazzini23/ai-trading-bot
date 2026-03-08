#!/usr/bin/env bash
set -euo pipefail

# High-signal strict type checks for critical runtime paths.
python3 -m mypy --config-file mypy_strict.ini \
  ai_trading/config/management.py \
  ai_trading/config/alpaca.py \
  ai_trading/config/__init__.py \
  ai_trading/config/settings.py \
  ai_trading/config/runtime.py \
  ai_trading/validation/validate_env.py \
  ai_trading/validation/require_env.py \
  ai_trading/health_payload.py \
  ai_trading/health.py \
  ai_trading/logging/emit_once.py \
  ai_trading/strategy_allocator.py \
  ai_trading/rl/module.py

# Runtime-critical modules that are not yet strict-clean still get explicit
# baseline type coverage in this gate.
python3 -m mypy --config-file mypy.ini \
  --follow-imports=skip \
  ai_trading/__main__.py \
  ai_trading/alpaca_api.py \
  ai_trading/app.py \
  ai_trading/core/bot_engine.py \
  ai_trading/data/fallback/concurrency.py \
  ai_trading/data/fetch/fallback_concurrency.py \
  ai_trading/data/universe.py \
  ai_trading/data/provider_monitor.py \
  ai_trading/execution/engine.py \
  ai_trading/execution/live_trading.py \
  ai_trading/settings.py \
  ai_trading/main.py \
  ai_trading/model_loader.py \
  ai_trading/logging/__init__.py \
  ai_trading/logging_filters.py \
  ai_trading/http/pooling.py \
  ai_trading/paths.py \
  ai_trading/net/http.py \
  ai_trading/data/fetch/__init__.py \
  ai_trading/policy/compiler.py \
  ai_trading/utils/env.py \
  ai_trading/utils/environment.py \
  ai_trading/utils/exec.py \
  ai_trading/env/__init__.py \
  ai_trading/util/env_check.py
