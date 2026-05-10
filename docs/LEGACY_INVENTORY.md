# Legacy Inventory

This inventory tracks retained legacy or operator-adjacent surfaces so they are
not mistaken for canonical runtime implementations.

## Retained Thin Surfaces

- Legacy demo and one-off validation scripts that seed dummy/test credentials
  are archival only. They are not production smoke tests and must not be used
  as deployment authority. Scripts that still seed dummy credentials,
  deprecated `ALPACA_BASE_URL`, or old `FLASK_PORT` defaults must call
  `scripts/legacy_guard.py` before seeding values and require
  `--allow-legacy-demo` or `AI_TRADING_ENABLE_LEGACY_DEMO=1`.
- `ai_trading.production_system` and
  `ai_trading.execution.production_engine` are retained for research/test
  harnesses only. They are not package-level public API exports and have no
  live execution authority; paper/live execution must use canonical
  OMS/pretrade.
- `scripts/health_check.py`: legacy CLI wrapper around the current health
  monitor. It returns a non-zero process status only when the overall status is
  critical.
- `scripts/debug_cli.py`: diagnostic CLI for execution state. It must use the
  caller's configured environment and must not seed dummy credentials or ports.
- `scripts/sync_env_runtime.sh`: packaged env renderer used by systemd. Packaged
  destinations under `/run/ai-trading-bot` fail closed when the runtime
  directory is unavailable.
- `scripts/runtime_env_sync.py`: canonical renderer for the runtime
  EnvironmentFile. It omits managed secret values when a secrets backend is in
  use.

## Confirm-First Operator Scripts

- `scripts/runtime_artifacts_reset.sh`: dry-run by default; requires
  `--confirm` to archive or rewrite runtime artifacts.
- `scripts/rollout_advanced_gates.sh`: dry-run by default; requires
  `--confirm` to edit `.env`, restart, or verify a rollout stage.

## Artifact And Docs Rules

- Makefile parsing must not create artifact directories or reports.
- Packaged health probes use `http://127.0.0.1:9001/healthz`.
- Standalone health on `HEALTHCHECK_PORT` is only for
  `RUN_HEALTHCHECK=1 python -m ai_trading.app` and must not be presented as the
  packaged service default.
- Hugging Face artifacts are research inputs only. They must remain outside
  production model paths and carry no runtime, promotion, or live-money
  authority.
