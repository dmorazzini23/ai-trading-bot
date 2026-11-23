# ai_trading Bug Hunt Phase 1 Manifest

Date: 2026-04-28

Scope: every tracked file under `ai_trading/`.

Phase 1 status: inventory and ownership complete. No bug hunting, code edits, or validation were performed in this phase.

## Summary

Total tracked `ai_trading/` files: 473

Every file starts Phase 2 with review status `unreviewed`. The table below is the single-owner ledger used to route detailed inspection. Subagent raw inventories were reconciled so each tracked file belongs to exactly one owner domain.

| Owner domain | Files | P0 | P1 | P2 | Initial status |
| --- | ---: | ---: | ---: | ---: | --- |
| live-money-execution-risk | 112 | 66 | 39 | 7 | unreviewed |
| data-model-correctness | 125 | 21 | 81 | 23 | unreviewed |
| runtime-ops-health-config | 105 | 24 | 45 | 36 | unreviewed |
| analytics-governance-market | 57 | 16 | 19 | 22 | unreviewed |
| foundation-shared-platform | 74 | 0 | 32 | 42 | unreviewed |

Priority meanings:

- P0: direct production safety, live trading correctness, startup/health/config integrity, broker/order/risk contracts, model artifact/data leakage surfaces, or core governance.
- P1: important correctness, orchestration, analytics, monitoring, portfolio/model behavior, or shared primitives with meaningful production impact.
- P2: package plumbing, support utilities, defaults, simulations, local tools, low-directness helpers, or lower-risk surfaces.

## Owner Domains

### live-money-execution-risk

Primary scope:

- `ai_trading/broker/`
- `ai_trading/execution/`
- `ai_trading/oms/`
- `ai_trading/risk/`
- `ai_trading/portfolio/`
- `ai_trading/position/`
- `ai_trading/order/`
- live-money top-level files: `alpaca_api.py`, `position_sizing.py`, `production_system.py`, `rebalancer.py`, `trade_logic.py`
- execution/risk-heavy `ai_trading/core/` files, including Alpaca client, bot engine, execution flow, netting, pending orders, position risk, and run-all-trades modules

Review focus for Phase 2:

- broker SDK contract correctness
- order side semantics for long, short, cover, flatten, cancel, and bracket/protective orders
- fail-closed order submission and reconciliation behavior
- gross/net exposure and cap enforcement
- symbol-specific position and available-quantity handling
- idempotency, duplicate-order, and lifecycle invariants

### data-model-correctness

Primary scope:

- `ai_trading/data/`
- `ai_trading/data_validation/`
- `ai_trading/features/`
- `ai_trading/indicators/`
- `ai_trading/talib/`
- `ai_trading/training/`
- `ai_trading/evaluation/`
- `ai_trading/research/`
- `ai_trading/ml/`
- `ai_trading/models/`
- `ai_trading/prediction/`
- `ai_trading/meta/`
- `ai_trading/meta_learning/`
- `ai_trading/rl/`
- `ai_trading/rl_trading/`
- `ai_trading/regime/`
- `ai_trading/strategies/`
- `ai_trading/pipeline/`
- `ai_trading/plotting/`
- data/model top-level files: `algorithm_optimizer.py`, `indicator_manager.py`, `ml_model.py`, `model_loader.py`, `model_registry.py`, `portfolio_rl.py`, `predict.py`, `simple_models.py`

Review focus for Phase 2:

- future leakage in indicators, labels, features, CV, and model selection
- timestamp/index preservation and alignment
- empty provider responses and malformed data handling
- model artifact metadata, checksums, and load/serve contracts
- strategy signal validity and same-bar execution assumptions
- invalid price/NaN/inf handling across training and RL paths

### runtime-ops-health-config

Primary scope:

- `ai_trading/app.py`
- `ai_trading/health.py`
- `ai_trading/health_monitor.py`
- `ai_trading/health_payload.py`
- `ai_trading/config/`
- `ai_trading/env/`
- `ai_trading/startup/`
- `ai_trading/runtime/`
- `ai_trading/diagnostics/`
- `ai_trading/logging/`
- `ai_trading/monitoring/`
- `ai_trading/telemetry/`
- `ai_trading/metrics/`
- `ai_trading/services/`
- `ai_trading/scheduler/`
- `ai_trading/safety/`
- `ai_trading/validation/`
- `ai_trading/tools/`
- `ai_trading/scripts/`
- `ai_trading/integrations/`
- `ai_trading/http/`
- `ai_trading/net/`
- `ai_trading/production/`
- runtime top-level files: `logging_filters.py`, `main.py`, `main_extended.py`, `main_trade_log_path.py`, `paths.py`, `process_manager.py`, `runner.py`, `security.py`, `settings.py`, `shutdown_handler.py`

Review focus for Phase 2:

- canonical health route registration and graceful degradation
- fail-fast env/config validation and managed env precedence
- secret redaction and structured logging behavior
- runtime artifact and path safety
- service orchestration, startup, shutdown, and scheduler behavior
- metrics/telemetry/monitoring contracts that affect operational truth

### analytics-governance-market

Primary scope:

- `ai_trading/analysis/`
- `ai_trading/analytics/`
- `ai_trading/backtesting/`
- `ai_trading/contracts/`
- `ai_trading/market/`
- `ai_trading/governance/`
- `ai_trading/guards/`
- `ai_trading/database/`
- `ai_trading/defaults/`
- `ai_trading/institutional/`
- `ai_trading/policy/`
- `ai_trading/price_snapshot/`
- `ai_trading/replay/`
- `ai_trading/retrain/`
- `ai_trading/shadow_mode/`
- `ai_trading/slippage/`
- `ai_trading/tca/`
- analytics/governance top-level files: `audit.py`, `capital_scaling.py`, `operator_presets.py`, `strategy_allocator.py`

Review focus for Phase 2:

- calendar and market-session correctness
- governance approval, rollout, replay, and parity gates
- analytics and TCA truthfulness
- database/model contract safety
- backtest/replay assumptions that can mask live bugs
- policy/compiler and guard behavior

### foundation-shared-platform

Primary scope:

- `ai_trading/__init__.py`
- `ai_trading/__main__.py`
- `ai_trading/exc.py`
- `ai_trading/exception_family.py`
- `ai_trading/hyperparams.json`
- `ai_trading/timeframe.py`
- non-execution `ai_trading/core/` shared primitives
- `ai_trading/util/`
- `ai_trading/utils/`
- `ai_trading/math/`
- `ai_trading/registry/`

Review focus for Phase 2:

- import-time side effects
- exception taxonomy and retry safety
- no-shim invariant
- timezone and timeframe correctness
- utility behavior that hides failures or mutates global process state
- serialization, pickle, subprocess, worker, and path helper safety

## Subagent Phase 1 Inputs

Five subagents performed read-only inventory slices:

- live-money-execution-risk: 109 raw files before reconciliation
- data-model-correctness: 117 raw files before reconciliation
- runtime-ops-health-config: 101 raw files before reconciliation
- analytics-governance-market: 58 raw files before reconciliation
- foundation-shared-platform: 111 raw files before reconciliation

Raw slice counts intentionally overlapped at boundaries such as config, contracts, metrics, strategy allocation, execution simulation, and shared utilities. The reconciled owner ledger above is authoritative for Phase 2 routing.

## Phase 2 Starting State

All 473 tracked files are assigned and start as `unreviewed`.

Phase 2 should inspect P0 files first, then P1, then P2. Any discovered critical, high, or medium risk bug should move through:

1. `bug-found`
2. `fixed`
3. `targeted-test-passed`
4. `ready-for-final-validation`

Low-risk findings should be logged separately as `low-risk-deferred` unless they are adjacent to an active critical/high/medium fix.
