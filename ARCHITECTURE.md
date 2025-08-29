# Architecture Overview

## System
- Python 3.12 app running on a DigitalOcean droplet, supervised by `systemd`.
- Entry: `ai_trading/main.py`; core loop in `ai_trading/core/bot_engine.py`.
- Alpaca SDK (`alpaca-py`) is imported lazily; startup preflight aborts if the SDK is missing.
- Health & metrics via `python -m ai_trading.app` when `RUN_HEALTHCHECK=1`.
  - `/healthz` JSON and `/metrics` Prometheus served on the port specified by the `HEALTHCHECK_PORT` environment variable (default **9001**).

## Object Model
 - **TradingConfig**: static config (API keys, paths, thresholds).
   - Confidence gate: `AI_TRADING_CONF_THRESHOLD` sets minimum model confidence (default **0.75**).
 - **BotRuntime**: process runtime (cfg, params, tickers, model, broker clients, etc.).
   - Required fields: `cfg`, `params: dict`, `tickers: list[str]`, `model: Any (optional)`.

## Control Flow (happy path)
1. `main.py` → loads config → constructs `BotRuntime`.
2. `bot_engine.run_all_trades_worker(runtime, state)`:
   - Param validation, PDT check, data fetch health check.
   - Candidate screening: `screen_candidates(runtime, runtime.tickers)` → `screen_universe(..., runtime)`.
   - Regime: `check_market_regime(runtime, state)` → `detect_regime_state(runtime)`.
   - Model: `_load_primary_model(runtime)` (cached at `runtime.model`).
   - Planning/execution (risk, sizing, orders).
3. Heartbeat logs.

## Logging
- Structured JSON logs with fields: `ts`, `level`, `name`, `msg`, domain extras (e.g., `tickers`).
- No `print()`. Use existing logger factories within `ai_trading.logging`.

## Error Handling
- Catch **specific** exceptions; fail fast on config errors.
- Avoid silent fallbacks; prefer explicit “SKIP_*” logs when skipping a stage.
- No `try/except ImportError` in prod—dependencies are explicit.

## Data & Models
- Candidate tickers: `tickers.csv` if present, else built-in fallback `[SPY, AAPL, MSFT, AMZN, GOOGL]`.
- Model loader tries: `cfg.ml_model_path` / `cfg.model_path` (joblib/pickle), or `cfg.ml_model_module` / `cfg.model_module` (import module, `get_model(cfg)` or `Model(cfg)`).
- Cache on `runtime.model`.

## Coding Conventions
- `runtime` parameter is mandatory in hot paths.
- No `ctx`. If a function historically used it, pass `runtime`; optionally alias `ctx = runtime` as a **local** variable.
- No dynamic `exec`/`eval`.

## Deployment
- venv + pinned deps; `systemd` unit `ai-trading.service`.
- Restart flow:
  ```bash
  sudo systemctl restart ai-trading.service
  journalctl -u ai-trading.service -f | sed -n '1,200p'
  ```

## CLI
- `ai-trade`, `ai-backtest`, `ai-health`
  - `--dry-run`
  - `--once`
  - `--interval SECONDS`
  - `--paper` / `--live`
