# ai_trading Bug Hunt Phase 2 P0 Findings

Date: 2026-04-28

Scope: P0 files from `docs/AI_TRADING_BUG_HUNT_PHASE1_MANIFEST.md`.

Phase 2 status: P0 inspection complete, and P0 critical/high/medium findings repaired. Six subagents inspected disjoint P0 slices, then six workers repaired disjoint ownership slices. No broad validation or full pytest was run because the repair occurred during regular U.S. market hours.

## Summary

P0 files inspected: 127

Findings by severity:

- Critical: 3
- High: 22
- Medium: 15

Total critical/high/medium findings: 40

Repair status:

- Critical repaired: 3 / 3
- High repaired: 22 / 22
- Medium repaired: 15 / 15
- Remaining known P0 critical/high/medium findings: 0

Recommended repair order:

1. Critical live trading/risk stop failures.
2. High live order, reconciliation, and risk-cap failures.
3. High health/config/secret leakage failures.
4. High data/model leakage and prediction failures.
5. High governance/market-session fail-open failures.
6. Medium correctness and contract issues.

## Critical Findings

- `ai_trading/production_system.py:125` - Market-data guarded executions risk-assess `integrated_recommendation.recommended_quantity`, which is initialized to `0` and never populated, then submit the caller's actual `quantity`. Live orders can bypass analysis/risk sizing. Fix by risk-checking the requested quantity and rejecting BUY/SELL when the approved/recommended quantity is not positive.

- `ai_trading/production_system.py:331` - Emergency shutdown halts only `ProductionTradingSystem.halt_manager`; the execution coordinator owns a separate halt manager, and `execute_trade()` does not check `is_active` or the system halt manager before submitting. Orders can still be submitted after stop/shutdown through this object. Fix by adding fail-closed active/halt gates and sharing halt state with the execution coordinator.

- `ai_trading/risk/circuit_breakers.py:92` - Drawdown circuit breaker can reopen trading while drawdown remains above the max threshold because reset uses `current_equity / peak_equity >= recovery_threshold`. Fix by requiring drawdown recovery below the halt threshold, preferably with hysteresis or manual reset.

## High Findings

- `ai_trading/core/run_all_trades_execution.py:177` - Native `alpaca-py` open orders are treated as unavailable because this gate only accepts `list_orders`, while `TradingClient` exposes `get_orders(filter=...)`. Pending-order blocking can fail open. Fix by accepting `get_orders` and routing through canonical open-order listing.

- `ai_trading/broker/adapters.py:92` - `AlpacaBrokerAdapter.list_orders()` calls native `get_orders(status=...)`, incompatible with `alpaca-py 0.42.1`. Open-order checks can undercount broker exposure. Fix by using `GetOrdersRequest` for native clients.

- `ai_trading/core/position_risk_runtime.py:73` - Short positions with positive `qty` plus `side="short"` are classified as long because signing uses only `qty`. Short trailing-stop/risk handling can be wrong. Fix by normalizing signed quantity from both `qty` and `side`.

- `ai_trading/execution/position_reconciler.py:76` - Broker reconciliation returns an empty broker position map without fetching the broker. Internal positions can be overwritten to zero. Fix by fetching real broker positions and failing closed on fetch errors.

- `ai_trading/execution/live_trading.py:2203` - Generated client order IDs use only symbol, side, and epoch minute. Legitimate same-minute orders can be suppressed as duplicates. Fix by including a full intent fingerprint or caller-supplied idempotency key.

- `ai_trading/execution/engine.py:3289` - A regular sell only checks `available_qty <= 0`; it does not reject `quantity > available_qty`, so the internal ledger can flip short without explicit short-sale intent. Fix by rejecting or clipping regular sells to available long quantity.

- `ai_trading/execution/production_engine.py:325` - Average price is wrong when a fill flips long to short or short to long; signed weighted averaging can produce impossible residual basis. Fix by splitting add/reduce/flat/flip cases and setting residual flip basis to fill price.

- `ai_trading/risk/engine.py:1068` - Risk hard stops automatically clear after a time cooldown without confirming drawdown recovery. Fix by requiring fresh recovered metrics or manual reset.

- `ai_trading/risk/pre_trade_validation.py:246` - Position/portfolio validation ignores order side, so short additions can look like exposure reductions. Fix by projecting signed position by side/intent and validating gross exposure.

- `ai_trading/risk/adaptive_sizing.py:300` - Adaptive multipliers can exceed concentration/max-position caps after base sizing already capped the position. Fix by reapplying final notional and exposure caps after multipliers.

- `ai_trading/training/after_hours.py:646` - Labels use the next available row, but `label_ts` is hard-coded to `timestamp + 1 day`; weekends/holidays/missing bars can break purge boundaries. Fix by storing the actual next valid bar timestamp.

- `ai_trading/model_loader.py:107` - Missing real data falls back to synthetic prices and can persist a production model from fake labels. Fix by failing closed outside explicit test/smoke mode and never writing synthetic production artifacts.

- `ai_trading/predict.py:49` - Prediction uses the first feature row and reports class-0 probability as confidence. This can serve stale/inverted predictions. Fix by scoring the latest row and selecting the positive-class probability.

- `ai_trading/health_payload.py:1683` - Health fallback can return HTTP 500 when payload/response construction fails, violating the health never-500 invariant. Fix by returning degraded `ok=false` payloads without 500.

- `ai_trading/app.py:639` - `create_app()` catches only `ImportError`/`RuntimeError` during startup validation; config parsing `ValueError` can crash before health routes are registered when `fail_fast_env=False`. Fix by catching the shared config/fallback exception family and caching `_ENV_ERR`.

- `ai_trading/config/scaling.py:89` - `ScalingConfig.from_env()` copies arbitrary unknown environment variables into `extras`, including raw secrets. Fix by allowlisting scaling keys or redacting/dropping sensitive unknowns.

- `ai_trading/validation/validate_env.py:135` - `validate_specific_env_var()` returns raw env values, including secrets, to callers. Fix by returning redacted values or non-secret metadata only.

- `ai_trading/market/calendars.py:121` - Futures/forex session truth is wrong for overnight/equal start-end sessions; probes showed false opens/closes. Fix by modeling day-specific open/close rules and explicit 24h sessions.

- `ai_trading/guards/staleness.py:78` - Future-dated bars pass freshness checks because negative age is accepted. Fix by rejecting timestamps ahead of `now` beyond a small skew allowance.

- `ai_trading/strategy_allocator.py:157` - Unreliable fallback prices can become reliable without a measured gap when provider is `yahoo`. Fix by requiring finite gap evidence or an audited override.

- `ai_trading/governance/promotion.py:1440` - Required promotion approval can pass with malformed or missing approval timestamp. Fix by rejecting missing/unparseable approval timestamps.

- `ai_trading/governance/promotion.py:83` - TCA gate fails open when telemetry is absent because `tca_gate_passed` defaults to true and eligibility preserves it. Fix by defaulting to false/unknown and requiring explicit passing TCA evidence.

## Medium Findings

- `ai_trading/core/bot_engine.py:19552` - `_enforce_buying_power_limit()` returns original quantity when available buying/shorting power is known and `<= 0`. Fix by returning zero for opening buys/shorts when capacity is non-positive.

- `ai_trading/broker/adapters.py:147` - Unknown order sides map to sell. A typo or unsupported alias can become a sell instead of failing closed. Fix by explicitly validating accepted side aliases and raising on unknown values.

- `ai_trading/execution/classes.py:137` - Default order/request IDs use second-level timestamps, causing possible same-second collisions. Fix by using UUID/ULID, nanoseconds plus counter, or required caller idempotency keys.

- `ai_trading/risk/adaptive_sizing.py:341` - Correlation penalty uses signed total notional, so offsetting long/short books can erase gross correlated exposure. Fix by using absolute notional for denominator and weights.

- `ai_trading/risk/engine.py:1218` - Final position size can exceed cash/position caps due to minimum quantity floors and a default 10-share floor. Fix by enforcing final notional/cash/exposure caps after minimum logic.

- `ai_trading/model_registry.py:352` - External artifact registration stores `external_path`, but `load_model()` returns `None` instead of loading/verifying the artifact. Fix by loading through verified joblib artifact helpers and checksum validation.

- `ai_trading/data/fetch/normalize.py:269` - OHLCV normalization can pass incomplete or invalid price data downstream. Fix by requiring all OHLCV columns, numeric finite positive prices, and OHLC consistency.

- `ai_trading/data/splits.py:245` - `validate_no_leakage()` logs temporal overlap but still returns `True`. Fix by returning false on overlap and enforcing embargo/horizon checks.

- `ai_trading/training/stacking.py:86` - Meta-label thresholding can create a one-class target and crash or create unusable gating. Fix by checking target cardinality and using an explicit constant/Dummy classifier or skipping gating.

- `ai_trading/validation/validate_env.py:153` - Env validation CLI exits successfully with missing required credentials. Fix by returning nonzero unless an explicit dry-run/test mode is set.

- `ai_trading/config/runtime.py:2620` - `TradingConfig.update()` mutates `_values` in place despite config being presented as immutable. Fix by returning a new config or adding locked/internal-only mutation with cache invalidation.

- `ai_trading/governance/rollout.py:291` - Capital ramp can upgrade on missing live telemetry because missing fields default to healthy zeroes. Fix by distinguishing missing telemetry and holding/breaching the gate.

- `ai_trading/market/symbol_specs.py:125` - Runtime symbol-spec updates do not refresh exported lookup maps. Fix by updating maps or resolving dynamically.

- `ai_trading/database/connection.py:207` - Nested sessions in one thread overwrite each other in `_connections`. Fix by using unique session IDs or per-thread session stacks.

- `ai_trading/contracts/decisioning.py:509` - Decision journals can mark rejected/vetoed intents as submitted. Fix by deriving submitted status from actual broker submission/status evidence.

## Phase 2 Notes

Read-only static sweeps were also run over P0 surfaces for Alpaca method contracts, leakage patterns, direct env/secret handling, and governance/session keywords. Candidate hits were either represented above or left for later P1/P2 review if not P0-critical.

## Repair Validation

Targeted validation after integrated repairs:

- `./venv/bin/pytest -q` on the repaired P0 surface: 461 passed.
- `./venv/bin/ruff check $(git diff --name-only -- '*.py')`: passed.
- `./venv/bin/python -m py_compile $(git diff --name-only -- '*.py')`: passed.
- `git diff --check`: passed.
- Targeted `mypy --follow-imports=skip` on changed P0 modules except legacy `ai_trading/core/bot_engine.py`: passed for 34 source files.

Known validation limitation:

- Full `./venv/bin/pytest -q`, full `./venv/bin/mypy`, `bash scripts/typecheck_strict.sh`, and full repo `py_compile` were not run during market hours.
- Including `ai_trading/core/bot_engine.py` in targeted mypy still reports legacy `no-any-return` errors away from the touched `_enforce_buying_power_limit()` change. The focused tests, ruff, py_compile, and integration-targeted suite passed.

Next recommended action: proceed to Phase 2 P1 inspection/repair after reviewing this P0 patchset, then run full validation outside market hours or with explicit approval.
