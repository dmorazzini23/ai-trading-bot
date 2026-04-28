# AI Trading Bug Hunt - Phase 2 P1 Findings

Date: 2026-04-28

Scope: P1 tracked files under `ai_trading/` from the reconciled Phase 1 manifest.

This was an inspection-only pass. Subagents were instructed not to run full validation during market hours and to use targeted investigation only. No runtime/library files were changed during this pass.

## Summary

- Files inspected: 216
- Findings: 51
- Critical: 2
- High: 22
- Medium: 27

## Repair Wave 1 - 2026-04-28

Status: completed during market hours with targeted validation only.

Fixed in this wave:

- Critical: 2 of 2
- High: 15 of 22
- Medium: 0 of 27

Resolved findings:

- Critical: reversed split adjustment in `ai_trading/data/corp_actions.py`
- Critical: one Alpaca success clearing provider safe mode in `ai_trading/data/provider_monitor.py`
- High: intraday bar requests expanding to full UTC days in `ai_trading/data/bars.py`
- High: unauthorized alternate daily feed blocking minute-resample fallback in `ai_trading/data/bars.py`
- High: non-terminal empty fetch attempts poisoning `_SKIPPED_SYMBOLS` in `ai_trading/data/fetch/backoff.py`
- High: stale dynamic-universe screener `_LAST_GOOD` reuse in `ai_trading/data/alpaca_screener.py`
- High: non-finite strategy strength/confidence becoming max-strength trades in `ai_trading/strategies/base.py`
- High: non-finite execution-simulation inputs corrupting metrics in `ai_trading/evaluation/execution_sim.py`
- High: NewsAPI rate-limit fallback bypassing production fail-closed sentiment in `ai_trading/analysis/sentiment.py`
- High: replay same-bar fill leakage and ignored OMS-gate config in `ai_trading/replay/replay_engine.py`
- High: sell-side slippage sign inversion in `ai_trading/tca/event_analytics.py` and `ai_trading/slippage/recorder.py`
- High: CLI main loop swallowing nonzero `SystemExit` in `ai_trading/__main__.py`
- High: AUTO max-position sizing failing open in `ai_trading/core/runtime.py`
- High: startup stale-data checks failing open by default in `ai_trading/core/startup_runtime.py`
- High: signal handler unsafe `asyncio.create_task` use in `ai_trading/shutdown_handler.py`
- High: embedded secrets leaking through formatted logs in `ai_trading/logging_filters.py`
- High: emergency safety actions/callbacks failing open in `ai_trading/safety/monitoring.py`

Repair validation:

- `./venv/bin/pytest -q ...changed targeted tests...` passed: 113 passed
- `./venv/bin/python -m py_compile ...changed Python files...` passed
- `./venv/bin/ruff check ...changed Python files...` passed
- `./venv/bin/mypy ...changed runtime modules...` failed on an existing imported-file error outside this repair wave: `ai_trading/execution/production_engine.py:344` incompatible assignment

Full repo validation was not run because the repair wave occurred during market hours.

Remaining High findings after this wave:

- Liquidity participation gate can fail open in `ai_trading/core/netting_symbol_cycle.py` and `ai_trading/core/netting_execution_context.py`
- RL training/inference feature mismatch in `ai_trading/rl_trading/train.py` and `ai_trading/rl_trading/inference.py`
- RL governance underreports return/drawdown in `ai_trading/rl_trading/train.py`
- Meta-learning signal polarity can be wrong with missing classes in `ai_trading/strategies/metalearning.py`
- Meta-learner labels reversed for non-normalized buy sides in `ai_trading/meta_learning/core.py`
- Meta-learning persistence sidecar is write-only in `ai_trading/meta_learning/persistence.py`
- Canary rollback and kill-switch paths use CWD defaults in `ai_trading/monitoring/model_liveness.py`

## Ownership Coverage

| Area | P1 files inspected | Critical | High | Medium | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Live-money execution and risk | 39 | 0 | 1 | 4 | 5 |
| Data provider correctness | 81 | 2 | 4 | 3 | 9 |
| Modeling and strategy correctness | 0 | 0 | 7 | 5 | 12 |
| Runtime, ops, health, and config | 45 | 0 | 4 | 2 | 6 |
| Analytics, governance, and market replay | 19 | 0 | 3 | 7 | 10 |
| Foundation and shared platform | 32 | 0 | 3 | 6 | 9 |

Note: the data/model P1 bucket was split between data/provider inspection and modeling/strategy inspection for review quality. The modeling/strategy rows are part of the 81-file data/model assignment, not additional tracked files.

## Critical Findings

### 1. Reversed Split Adjustment Corrupts Historical Bars

- File: `ai_trading/data/corp_actions.py:189`
- Severity: Critical
- Impact: `adjust_bars()` can multiply pre-split prices by the split ratio instead of dividing, corrupting features, labels, sizing, and backtests.
- Evidence: split factor is stored as `1.0 / ratio`; `get_adjustment_factors()` inverts it when `reference_date > target_date`; `adjust_bars()` applies that inverted factor to older rows.
- Suggested fix: remove the historical-to-current inversion or split the API into explicit adjust and reverse-adjust paths. Add a known split regression test for both price and volume.

### 2. Provider Safe Mode Clears After One Alpaca Success

- File: `ai_trading/data/provider_monitor.py:1864`
- Severity: Critical
- Impact: A single Alpaca success can clear in-memory safe mode immediately, bypassing recovery pass count, freshness, and cooldown protections. Trading can resume too early after a provider kill switch.
- Evidence: `record_success()` sets `_SAFE_MODE_ACTIVE=False`; the gated recovery path is `_maybe_clear_safe_mode()`.
- Suggested fix: do not clear safe mode in `record_success()`. Route recovery through the existing health-pass/cooldown mechanism.

## High Findings

### 1. Liquidity Participation Gate Can Fail Open

- Files: `ai_trading/core/netting_symbol_cycle.py:437`, `ai_trading/core/netting_execution_context.py:371`
- Severity: High
- Impact: Liquidity participation caps can be auto-disabled because the gate is not in the critical gate set. The bypass path records `LIQ_PARTICIPATION_BLOCK_BYPASSED` and continues with the requested delta.
- Suggested fix: make liquidity participation and cap gates non-disableable critical gates. In block mode, never continue after `allowed_participation=False`.

### 2. Intraday Bar Requests Expand To Full UTC Days

- File: `ai_trading/data/bars.py:794`
- Severity: High
- Impact: Minute requests can fetch 00:00-23:59 UTC instead of the requested window, polluting intraday features with unintended session data.
- Evidence: `safe_get_stock_bars()` clamps start/end to beginning/end of day before timeframe-specific handling.
- Suggested fix: only day-clamp daily requests. Preserve exact intraday timestamps or clamp intraday windows to the exchange session.

### 3. Unauthorized SIP Fallback Can Block Minute Resampling

- File: `ai_trading/data/bars.py:1053`
- Severity: High
- Impact: Daily fallback blindly tries SIP when IEX is empty. If the account lacks SIP entitlement, the unauthorized alternate feed can abort before minute-resample fallback runs.
- Suggested fix: choose alternate feeds through entitlement checks, and let unauthorized alternate-feed failures fall through to minute resampling.

### 4. Empty Fetch Attempt Can Poison Future Provider Calls

- File: `ai_trading/data/fetch/backoff.py:220`
- Severity: High
- Impact: One empty attempt can add a symbol/timeframe to `_SKIPPED_SYMBOLS`, causing later calls to return empty before retrying providers.
- Suggested fix: separate in-flight tracking from max-retry skip state, or clear the skip marker whenever the call exits without a terminal max-retry empty-bars failure.

### 5. Dynamic Universe Can Reuse Stale Screener Results

- File: `ai_trading/data/alpaca_screener.py:284`
- Severity: High
- Impact: After screener failures, market movers/actives can be reused indefinitely because `_LAST_GOOD` fallback does not check age or market day.
- Suggested fix: store fetch timestamp/session with `_LAST_GOOD`; require same market day and max age before returning stale candidates.

### 6. NaN Strategy Strength Becomes Maximum Strength

- File: `ai_trading/strategies/base.py:52`
- Severity: High
- Impact: Non-finite signal strength or confidence can convert to `1.0`, turning invalid strategy output into max-confidence trades.
- Suggested fix: reject or neutralize non-finite values before clamping. Add finite checks in signal validation.

### 7. Non-Finite Simulation Inputs Corrupt Evaluation Metrics

- File: `ai_trading/evaluation/execution_sim.py:107`
- Severity: High
- Impact: NaN or infinite predictions/returns can propagate into fold equity and metrics, corrupting validation gates.
- Suggested fix: filter or neutralize non-finite rows, track invalid-row counts, and fail metrics above an explicit tolerance.

### 8. RL Training And Inference Use Different Feature Semantics

- Files: `ai_trading/rl_trading/train.py:856`, `ai_trading/rl_trading/inference.py:175`
- Severity: High
- Impact: Training uses state-builder normalized features, while inference pads raw observations. The resulting train/inference mismatch can produce unsafe actions.
- Suggested fix: persist the full state-builder schema/statistics with the model, load and apply them at inference, and fail on metadata mismatch.

### 9. RL Governance Underreports Return And Drawdown

- File: `ai_trading/rl_trading/train.py:1017`
- Severity: High
- Impact: Episode-level net return and max drawdown are divided by step count, which can incorrectly pass promotion gates.
- Suggested fix: average only additive per-step metrics. Keep terminal return and max drawdown as episode-level metrics before cross-episode aggregation.

### 10. Meta-Learning Signal Polarity Can Be Wrong With Missing Classes

- Files: `ai_trading/strategies/metalearning.py:246`, `ai_trading/strategies/metalearning.py:329`
- Severity: High
- Impact: If training data contains only two of the expected three labels, probability columns can be mapped to the wrong trade direction.
- Suggested fix: map probabilities by `model.classes_`, and require or explicitly handle the expected label set.

### 11. Meta-Learner Reverses Labels For Non-Normalized Buy Sides

- File: `ai_trading/meta_learning/core.py:1174`
- Severity: High
- Impact: Uppercase or alternate buy-side values such as `BUY`, `Buy`, or `long` can be treated as short.
- Suggested fix: normalize side values and reject unknown sides rather than defaulting them.

### 12. Meta-Learning Persistence Sidecar Is Write-Only

- Files: `ai_trading/meta_learning/persistence.py:318`, `ai_trading/meta_learning/persistence.py:367`
- Severity: High
- Impact: Without a parquet engine, trade history may be written to a pickle sidecar that reads never load, silently dropping history.
- Suggested fix: either fail writes when parquet support is missing or explicitly read and migrate the trusted sidecar path.

### 13. NewsAPI Rate Limit Bypasses Production Fail-Closed Sentiment

- File: `ai_trading/analysis/sentiment.py:317`
- Severity: High
- Impact: NewsAPI 429 handling can return and cache neutral sentiment instead of honoring production fail-closed sentiment semantics.
- Evidence: the enhanced rate-limit handler calls `_get_cached_or_neutral_sentiment` without its required `reason`, catches the resulting `TypeError`, and returns `0.0`.
- Suggested fix: pass the required reason and route final fallback through the canonical cached/fail-closed helper.

### 14. Replay Engine Ignores Fill Timing And OMS Gate Configuration

- File: `ai_trading/replay/replay_engine.py:28`
- Severity: High
- Impact: Replay fills orders immediately on the current bar and does not honor configured fill simulation or OMS gates, masking live execution failures and same-bar leakage.
- Suggested fix: route orders through pending fill simulation and gate enforcement when configured, filling only on the next eligible bar.

### 15. Sell-Side Slippage Sign Is Inverted

- Files: `ai_trading/tca/event_analytics.py:339`, `ai_trading/slippage/recorder.py:54`
- Severity: High
- Impact: Sell-side adverse and favorable slippage can be reported with the wrong sign.
- Suggested fix: compute side-normalized signed basis points before aggregation and persistence.

### 16. CLI Main Loop Swallows Fatal SystemExit

- File: `ai_trading/__main__.py:110`
- Severity: High
- Impact: Fatal exits can be logged and converted to process exit code 0.
- Suggested fix: re-raise nonzero `SystemExit` or propagate the exit code to the process boundary.

### 17. AUTO Max-Position Sizing Can Fail Open

- File: `ai_trading/core/runtime.py:157`
- Severity: High
- Impact: AUTO sizing failures can be converted into static fallback sizing, allowing trading to continue with default position limits after live equity sizing aborts.
- Suggested fix: in AUTO mode, let sizing aborts propagate. Limit fallback behavior to explicit static sizing modes.

### 18. Startup Health Allows Stale Data By Default

- File: `ai_trading/core/startup_runtime.py:313`
- Severity: High
- Impact: Runtime can start and trade on stale symbols unless `ALLOW_STALE_DATA_STARTUP=false` is explicitly configured.
- Suggested fix: fail closed by default, with an explicit deployment override for warmup scenarios.

### 19. Signal Handler Can Fail To Schedule Graceful Shutdown

- File: `ai_trading/shutdown_handler.py:95`
- Severity: High
- Impact: SIGTERM/SIGINT can raise `RuntimeError` if no event loop is running in the signal-handling thread, leaving shutdown cleanup unscheduled.
- Suggested fix: have the signal path set a shutdown request flag/event, then schedule shutdown from a known running loop or worker thread with fallback handling.

### 20. Canary Rollback And Kill-Switch Paths Use CWD Defaults

- File: `ai_trading/monitoring/model_liveness.py:359`
- Severity: High
- Impact: Rollback and kill-switch files can be written under process cwd rather than the managed runtime directory, so ops automation may miss them or systemd sandboxing may reject writes.
- Suggested fix: resolve these paths through canonical runtime artifact helpers and align them with runtime safety monitoring paths.

### 21. Log Redaction Misses Embedded Secrets

- File: `ai_trading/logging_filters.py:83`
- Severity: High
- Impact: Secrets embedded inside formatted arguments can leak because only exact argument matches are redacted.
- Suggested fix: redact candidate substrings inside string args and/or redact the final formatted message before JSON serialization.

### 22. Emergency Safety Actions Can Fail Open

- File: `ai_trading/safety/monitoring.py:102`
- Severity: High
- Impact: A `RuntimeError`, API exception, or `OSError` from one emergency action can prevent later emergency actions and alert callbacks from running.
- Suggested fix: isolate each action/callback with the repo's operational exception handling, log structured failure details, and continue.

## Medium Findings

### 1. Optimizer Init Failure Does Not Block Live Submit Path

- Files: `ai_trading/core/netting_target_runtime.py:393`, `ai_trading/core/netting_submit_prelude.py:79`
- Severity: Medium
- Impact: If execution portfolio optimizer is enabled but initialization fails, live orders can proceed without the configured gate.
- Suggested fix: carry optimizer init-failed state into submit prelude and block with `PORTFOLIO_OPTIMIZER_INIT_FAILED` unless an explicit fail-open setting is enabled.

### 2. Portfolio Max Weight Can Be Exceeded After Normalization

- File: `ai_trading/portfolio/sizing.py:284`
- Severity: Medium
- Impact: Small universes can exceed per-symbol `max_weight` after weights are normalized.
- Suggested fix: use capped redistribution/water-fill semantics or leave unused cash/gross exposure.

### 3. Portfolio Optimizer Mixes Return And Transaction Cost Units

- Files: `ai_trading/portfolio/optimizer.py:207`, `ai_trading/portfolio/optimizer.py:220`
- Severity: Medium
- Impact: Share-return units are compared with dollar transaction costs, distorting optimization and gating.
- Suggested fix: compute expected return changes and transaction costs in the same unit, preferably dollars or basis points.

### 4. Rebalancer Loses Short Direction

- Files: `ai_trading/rebalancer.py:223`, `ai_trading/rebalancer.py:459`, `ai_trading/rebalancer.py:464`
- Severity: Medium
- Impact: Short holdings can be treated as long holdings because quantity sign is discarded.
- Suggested fix: preserve signed quantities and emit buy-to-cover or sell-short semantics where appropriate.

### 5. Yahoo Fallback Cache Can Store Empty Or Malformed Data

- File: `ai_trading/data/fetch_yf.py:323`
- Severity: Medium
- Impact: Empty or malformed downloads can be cached for the day before OHLCV validation runs.
- Suggested fix: validate per-symbol OHLCV before cache writes.

### 6. Reference Reconcile Double-Counts New Rows

- File: `ai_trading/data/reference_reconcile.py:198`
- Severity: Medium
- Impact: Newly reconciled rows can be counted twice in reliability scores.
- Suggested fix: compute reliability from the reread dataset only, or score prior rows plus new rows before writing.

### 7. Timed-Out Fallback Workers Can Run Again

- File: `ai_trading/data/fallback/concurrency.py:1230`
- Severity: Medium
- Impact: Timed-out provider calls can be retried immediately outside the timeout budget, duplicating calls or extending hangs.
- Suggested fix: mark timed-out symbols failed for that call and avoid rerunning the same callable from the timeout handler.

### 8. Backtester Uses Same-Bar Fills

- File: `ai_trading/strategies/backtester.py:576`
- Severity: Medium
- Impact: Decisions and fills can occur at the same bar close, producing optimistic backtest results.
- Suggested fix: apply decisions from bar t to fills on bar t+1 with explicit latency semantics.

### 9. Legacy Backtest PnL Uses Random Outcomes

- File: `ai_trading/strategies/backtest.py:432`
- Severity: Medium
- Impact: Backtest metrics can be non-reproducible and detached from historical prices.
- Suggested fix: use deterministic historical next-bar or exit-rule prices. Keep stochastic stress testing separate.

### 10. Opening Short Signals Use Plain Sell

- Files: `ai_trading/strategies/cross_sectional_momentum.py:105`, `ai_trading/strategies/multi_factor_quality_value.py:134`, `ai_trading/strategies/pead_event.py:77`
- Severity: Medium
- Impact: Opening negative-alpha legs can be interpreted as liquidation rather than short entry.
- Suggested fix: reserve `sell` for exits and use `sell_short` for opening short alpha.

### 11. RL Inference Signals Can Be Dropped For Missing Strength

- Files: `ai_trading/rl_trading/inference.py:183`, `ai_trading/strategies/base.py:264`
- Severity: Medium
- Impact: RL actions may produce signals that fail validator checks because they lack explicit finite strength.
- Suggested fix: set strength from finite action magnitude or confidence.

### 12. Stacking Meta-Model Never Receives Usable Training Rows

- File: `ai_trading/strategies/signals.py:405`
- Severity: Medium
- Impact: Performance observations store `features: None`, so `_prepare_training_data()` skips them and the stacking model cannot learn.
- Suggested fix: persist the meta-features used at aggregation time and attach them when returns are recorded.

### 13. Governance Env Parsing Can Fail Open

- File: `ai_trading/policy/compiler.py:335`
- Severity: Medium
- Impact: Malformed governance env values can silently fall back to defaults, and invalid strict-mode tokens can become false.
- Suggested fix: use strict parsers for known forms and raise `PolicyConfigError` under strict governance.

### 14. Bad-Session Replay Can Leak Execution Outcomes Into Inputs

- File: `ai_trading/replay/bad_session.py:35`
- Severity: Medium
- Impact: Replay datasets can include fill/TCA outcome rows as market bars, leaking execution outcomes into replay inputs.
- Suggested fix: whitelist market/pre-trade input events and exclude fill/TCA rows.

### 15. Scoped TCA Outcome Counts Exclude Some Fills

- File: `ai_trading/tca/event_analytics.py:333`
- Severity: Medium
- Impact: Scoped outcome analytics undercount fills that lack `expected_price`.
- Suggested fix: count every fill event for scoped fill totals and gate only the slippage sample separately.

### 16. Cost Calibration Excludes Zero-Slippage Fills

- File: `ai_trading/tca/rollups.py:153`
- Severity: Medium
- Impact: Median costs and sample counts are biased because zero-slippage fills are excluded.
- Suggested fix: include every finite filled sample and expose nonzero/adverse samples separately if needed.

### 17. Missing Execution Timestamps Never Age Out

- File: `ai_trading/analytics/execution_report.py:299`
- Severity: Medium
- Impact: Phase 2 execution gates may not age out events with missing timestamps.
- Suggested fix: exclude invalid timestamps, report their count, and fail or mark indeterminate above a threshold.

### 18. Grid Runner Can Overwrite Same-Second Artifacts

- File: `ai_trading/backtesting/grid_runner.py:35`
- Severity: Medium
- Impact: Multiple parameter grid runs in the same second can overwrite artifacts.
- Suggested fix: use a unique run id with nanosecond timestamp, UUID, or config hash, and create artifact paths atomically without overwrite.

### 19. Form 4 Sentiment Component Is Dead Code

- File: `ai_trading/analysis/sentiment.py:556`
- Severity: Medium
- Impact: Insider-buy sentiment never contributes because filings are never appended.
- Suggested fix: either parse and append valid insider transactions or disable/remove the weight.

### 20. Mean-Reversion Polarity Is Inverted In Ensemble Voting

- File: `ai_trading/signals/__init__.py:693`
- Severity: Medium
- Impact: Positive mean-reversion z-scores can contribute buy votes even though they represent overbought conditions.
- Suggested fix: negate mean-reversion z-scores or add explicit polarity metadata before voting.

### 21. Static UTC Market Hours Are DST-Wrong

- File: `ai_trading/core/constants.py:10`
- Severity: Medium
- Impact: Static UTC open/close constants are wrong during daylight saving time.
- Suggested fix: derive exchange windows from an exchange-local calendar instead of fixed UTC constants.

### 22. TimeFrame Compatibility Layer Violates SDK Fail-Fast Policy

- File: `ai_trading/timeframe.py:1`
- Severity: Medium
- Impact: A fake `TimeFrame` fallback can hide missing or broken `alpaca-py` runtime dependencies.
- Suggested fix: fail fast in runtime imports. Confine SDK stubs to tests if needed.

### 23. Parameter Validator Logs At Import Time

- File: `ai_trading/core/parameter_validator.py:180`
- Severity: Medium
- Impact: A global `ParameterValidator` instance performs logging at import time.
- Suggested fix: lazy-create the validator through an accessor.

### 24. Manifest Validation Accepts Non-Finite Values

- File: `ai_trading/registry/manifest.py:50`
- Severity: Medium
- Impact: NaN, infinity, or unbounded threshold/cost values can pass validation.
- Suggested fix: require finite numeric values, probability thresholds in `[0, 1]`, and nonnegative costs.

### 25. Data Contract Silently Reinterprets Naive Timestamps

- File: `ai_trading/core/data_contract.py:65`
- Severity: Medium
- Impact: Naive timestamps are localized to caller-provided timezone, which can shift cached or provider bars by hours. The RTH predicate also admits the 16:00 boundary timestamp.
- Suggested fix: require timezone-aware bars or explicit source timezone metadata; use calendar session windows and treat RTH as ending before 16:00.

### 26. Runtime Prometheus Gauges Can Stay Stale

- File: `ai_trading/telemetry/runtime_prom_metrics.py:75`
- Severity: Medium
- Impact: Missing or unreadable runtime reports leave old gauge values visible indefinitely.
- Suggested fix: publish explicit stale/missing indicators, update report age, and clear or mark unavailable stale execution metrics.

### 27. Broker Reconciliation Errors Are Not Surfaced

- File: `ai_trading/services/reconciliation.py:67`
- Severity: Medium
- Impact: Broker reconciliation errors are logged but not exposed to callers, allowing stale stop/take-profit targets to remain after failed broker reads.
- Suggested fix: return an explicit failure result or raise a reconciliation exception and feed it into readiness/health paths.

## Validation

No full validation was run. This pass was inspection-only and occurred during market hours, so subagents were instructed to avoid broad validation and use targeted investigation only.
