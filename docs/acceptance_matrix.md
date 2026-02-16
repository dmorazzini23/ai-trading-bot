# Institutional Acceptance Matrix

This matrix maps institutional requirements to concrete tests and deployment checks.

| ID | Requirement | Code Anchor(s) | Test(s) | Deploy Check |
|---|---|---|---|---|
| AC-01 | Paper/live runtime contract fails fast without stubs | `ai_trading/core/runtime_contract.py`, `ai_trading/core/bot_engine.py` | `tests/test_runtime_contract_no_stubs.py` | `bash ci/scripts/institutional_gates.sh` runtime gate |
| AC-02 | Exception taxonomy and deterministic actions | `ai_trading/core/errors.py` | `tests/test_error_classification.py` | Gate script required test group |
| AC-03 | Dependency breakers open/close correctly | `ai_trading/core/dependency_breakers.py` | `tests/test_dependency_breakers.py` | Breaker-open threshold checks in monitoring |
| AC-04 | Retry only for idempotent reads | `ai_trading/core/retry.py` | `tests/test_retry_idempotent_reads_only.py` | Gate script required test group |
| AC-05 | Pretrade size/collar/duplicate controls enforced | `ai_trading/oms/pretrade.py` | `tests/test_pretrade_max_order_size.py`, `tests/test_pretrade_price_collar.py`, `tests/test_pretrade_duplicate_block.py` | Env key presence check + test pass |
| AC-06 | Kill switch optionally cancels all open orders | `ai_trading/oms/cancel_all.py`, `ai_trading/core/bot_engine.py` | `tests/test_kill_switch_cancel_all.py` | Runbook check (`runbooks/kill_switch_cancel_all.md`) |
| AC-07 | Run manifest written with startup contract details | `ai_trading/runtime/run_manifest.py` | `tests/test_run_manifest_written.py` | Path and writeability check |
| AC-08 | Trading mode not overwritten by execution mode flags | `ai_trading/__main__.py`, `ai_trading/config/runtime.py` | `tests/test_trading_mode_not_overwritten_by_cli.py` | Gate script required test group |
| AC-09 | Aggressive regime profile exists (no silent fallback) | `ai_trading/core/bot_engine.py` | `tests/test_regime_profile_aggressive_present.py` | Required test group |
| AC-10 | Liquidity mode naming consistent (`balanced`) | `ai_trading/execution/liquidity.py` | `tests/test_liquidity_mode_balanced_supported.py` | Required test group |
| AC-11 | TCA records and implementation shortfall produced | `ai_trading/analytics/tca.py` | `tests/test_tca_implementation_shortfall.py` | Daily report generation check |
| AC-12 | Daily execution report produced | `ai_trading/analytics/execution_report.py` | `tests/test_execution_report_daily_rollup.py` | Output dir check |
| AC-13 | Replay deterministic and broker submit blocked | `ai_trading/replay/replay_engine.py` | `tests/test_replay_engine_deterministic.py` | Replay gate in CI |
| AC-14 | Walk-forward + leakage checks fail hard on contamination | `ai_trading/research/walk_forward.py`, `ai_trading/research/leakage_tests.py` | `tests/test_walk_forward_no_leakage.py` | Leakage gate in CI |
| AC-15 | Order type config fail-fast enforced | `ai_trading/oms/orders.py`, `ai_trading/core/bot_engine.py` | `tests/test_order_types_supported_and_failfast.py`, `tests/test_order_type_startup_failfast.py` | Capability config preflight |
| AC-16 | Portfolio-level scaling/caps enforced | `ai_trading/risk/portfolio_limits.py` | `tests/test_portfolio_limits_vol_targeting.py` | Required test group |
| AC-17 | Dynamic allocation bounded updates | `ai_trading/portfolio/allocation.py` | `tests/test_allocation_weight_updates.py` | Required test group |
| AC-18 | Liquidity participation gates active | `ai_trading/risk/liquidity_regime.py` | `tests/test_liquidity_participation_block.py` | Required test group |
| AC-19 | Post-trade learning bounded and logged | `ai_trading/analytics/post_trade_learning.py` | `tests/test_post_trade_learning_bounded_updates.py` | Override file and delta checks |
| AC-20 | Quarantine deterministic and time-bounded | `ai_trading/runtime/quarantine.py` | `tests/test_quarantine_triggers_and_blocks.py` | Runbook + state path checks |
| AC-21 | Decision records include deterministic config snapshot | `ai_trading/core/netting.py`, `ai_trading/core/bot_engine.py` | `tests/test_decision_record_config_snapshot.py` | Snapshot toggle checks |
| AC-22 | Model artifacts verified, live fails closed | `ai_trading/models/artifacts.py`, `ai_trading/core/bot_engine.py` | `tests/test_model_artifacts.py`, `tests/test_model_verification_policy.py` | Manifest presence/verification in deploy checklist |
| AC-23 | Pretrade env key alignment for production `.env` keys | `ai_trading/oms/pretrade.py`, `ai_trading/core/bot_engine.py` | `tests/test_pretrade_env_key_alignment.py` | Gate script env sanity checks |

## Deployment Sign-Off Checklist
1. Institutional gate script passes.
2. Replay deterministic check passes for release seed/window.
3. Leakage checks pass.
4. Model manifest verification passes.
5. Canary symbol list and rollback thresholds configured.
6. Runbooks linked in release ticket and on-call assigned.
