# Robustness Audit

Last reviewed: `2026-04-27`

This document turns "make the system robust" into a concrete operating standard:

- each important failure mode has a prevention control
- each failure mode is visible through telemetry or alerts
- each failure mode has an automated drill or regression test
- each failure mode has an operator action path when it still occurs

The goal is not to prove the system can never fail. The goal is to keep failures
bounded, visible, and recoverable.

## Current Summary

Current status as of `2026-04-27`:

- `PASS`: backtester and offline replay are both implemented and validated.
- `PASS`: runtime ML now fails closed instead of silently using placeholder artifacts.
- `PASS`: after-hours runtime promotions now persist durable registry metadata.
- `PASS`: health, database, broker connectivity, and market-closed behavior are currently stable.
- `MONITOR`: the runtime is intentionally in explicit no-ML mode because no real promoted ML artifact exists yet.
- `MONITOR`: RL remains `shadow` gated and is not ready for live decisioning.
- `PASS`: strict pre-open flat-start enforcement is enabled in the local
  runtime config through `AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START=1`;
  keep `AI_TRADING_EXECUTION_PREOPEN_EXPECTED_SWING_SYMBOLS` current before
  allowing intentional swing exposure.
- `PASS`: `oms_invariants` and `oms_lifecycle_parity` are required by local
  health readiness through `AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS=1` and
  `AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY=1`.
- `PASS`: paper runtime now exercises the execution go/no-go gate and blocks
  degraded-data execution instead of widening into low-quality quotes.
- `ATTENTION`: live go/no-go is still blocked by observed performance evidence,
  not readiness plumbing.

## Control Matrix

| Failure mode | Prevention | Detection | Fallback / containment | Evidence | Current status |
| --- | --- | --- | --- | --- | --- |
| Stale / degraded market data | quote staleness thresholds, provider guardrails, backup routing, provider safe-mode | `/healthz`, provider telemetry, stale quote tests | halt flag, backup provider, shadow/degraded cycle | `tests/test_data_gating.py`, `tests/execution/test_live_trading_degraded_feed.py` | `PASS` |
| Broker unavailable / rejects / pacing pressure | broker capacity preflight, dependency breakers, pacing caps | broker status telemetry, execution TCA, reject-rate SLOs | halt / skip new orders, guarded retries | `tests/test_broker_capacity_preflight.py`, `tests/execution/test_execution_runtime_controls.py` | `PASS` |
| Reconciliation drift between local and broker state | reconciliation retry, mismatch burst tracking, opening freeze | go/no-go observed metrics, health payload positions section | freeze new openings, guarded auto-repair | `tests/execution/test_execution_runtime_controls.py` | `PASS` |
| Stale pending / orphaned orders | stale-order sweeps, startup stale-only cleanup, pending backlog caps | order health monitor, pending-order tests | cancel stale orders before new work | `tests/bot_engine/test_pending_orders_cleanup.py`, `tests/execution/test_execution_runtime_controls.py` | `PASS` |
| Market-open with unsafe runtime artifacts | pre-open readiness gates for broker/data/artifact freshness | readiness context, health/control-plane checks | block new openings until fresh | `tests/execution/test_execution_runtime_controls.py::test_runtime_preopen_readiness_*` | `PASS` |
| Non-flat account during closed session or before open | EOD flatten, startup reconciliation, enforced `AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START` guard with `AI_TRADING_EXECUTION_PREOPEN_EXPECTED_SWING_SYMBOLS` allowlist | readiness context, broker sync counts, health attention flags, journal logs | block new openings / operator flatten / hold rollout until expected | `tests/execution/test_execution_runtime_controls.py`, live service logs, `/healthz` broker counts | `PASS` |
| Symbol / gross / net exposure breach | risk engine caps, notional caps, sleeve caps, max trades/day | risk logs, runtime summaries, health/control-plane | block or scale targets, manage existing only | `config/runtime.py`, runtime risk checks | `PASS` |
| Hard dependency outage (provider / broker) | dependency breakers, safe-mode, halt flag | `/healthz`, provider state, alerting | fail safe and manage existing only | `docs/OPERATIONS.md`, provider safe-mode logic | `PASS` |
| Placeholder or missing ML artifact | explicit runtime loader verification, placeholder detection | `MODEL_RUNTIME_DISABLED`, loader tests | no-ML mode instead of fake model | `tests/test_model_loading.py` | `PASS` |
| Stale or missing production registry artifacts | viable production lookup, runtime promotion persistence | registry fallback tests, promotion metadata | use durable runtime-promoted path or disable ML | `tests/test_model_registry.py`, `tests/test_after_hours_training.py` | `PASS` |
| Weak live learning performance | promotion gates, confidence gate, live go/no-go | runtime performance reports, governance snapshots | keep shadow / no-ML, refuse promotion | runtime reports under `/var/lib/ai-trading-bot/runtime` | `MONITOR` |
| Health surface hides operator-relevant risk | canonical health payload, broker count exposure, attention flags | `/healthz`, `/operator/control-plane` | operator review before open | `tests/test_health_endpoints.py`, `tests/health/test_health_endpoint.py` | `PASS` |

## Scenario Drills

These are the minimum drills that should stay green.

### Data Integrity

- stale fallback quote blocks execution
- provider degradation triggers safe-mode / backup behavior
- duplicate and malformed historical bars are handled deterministically

Reference coverage:

- `tests/test_data_gating.py`
- `tests/execution/test_live_trading_degraded_feed.py`
- `tests/test_historical_bars.py`

### Broker / OMS

- broker unready before open blocks new openings
- stale pending orders are swept
- reconciliation mismatch burst freezes openings and can trigger guarded auto-repair
- partial-fill and stale pending TCA paths stay consistent

Reference coverage:

- `tests/execution/test_execution_runtime_controls.py`
- `tests/bot_engine/test_pending_orders_cleanup.py`
- `tests/test_tca_implementation_shortfall.py`

### Learning / Governance

- missing model files fail closed
- placeholder model files fail closed
- runtime registry fallback uses a viable promoted artifact when available
- shadow governance does not promote runtime artifacts
- production promotions write durable runtime metadata

Reference coverage:

- `tests/test_model_loading.py`
- `tests/test_model_registry.py`
- `tests/test_after_hours_training.py`

### Health / Operator Visibility

- market-closed warm state remains healthy when expected
- unknown broker/provider states do not get upgraded to healthy
- health payload exposes broker counts and market-closed attention flags

Reference coverage:

- `tests/test_health_endpoints.py`
- `tests/health/test_health_endpoint.py`

## Current Audit Findings

### 1. Pre-open account state has an explicit strict guard

The runtime can now require a flat start during the pre-open readiness window:

- `AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START=1` blocks opening trades when
  broker state shows open orders or unexpected nonzero positions.
- `AI_TRADING_EXECUTION_PREOPEN_EXPECTED_SWING_SYMBOLS` allows intentional swing
  holdings to pass while still blocking unexpected exposure.
- The readiness context includes a `flat_start` section with counts, expected
  and unexpected positions, and the concrete block reason.

The local runtime config now enables this strict guard. Do not add intentional
swing exposure without also updating `AI_TRADING_EXECUTION_PREOPEN_EXPECTED_SWING_SYMBOLS`
and validating that the broker snapshot reports the expected holdings.

### 2. Learning-system plumbing is now stronger than learning-system performance

The runtime now behaves honestly:

- ML runtime enters explicit no-ML mode when the configured artifact is only a placeholder
- RL runtime remains shadow-gated
- production fallback prefers viable promoted artifacts over stale registry pointers

What still blocks promotion is model performance, not deployment plumbing:

- live go/no-go must pass
- the consecutive-pass requirement must be met
- effective trade count must reach the configured threshold
- the current gate still rejects the runtime on live sample count and win rate

### 3. Operator visibility improved, and strict OMS gates are now configurable

The system now surfaces broker position/open-order counts and market-closed
attention flags in the canonical health payload. That helps operators catch
"non-flat before the open" early.

The OMS controls are explicit and enabled in the local runtime config:

- `AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS=1`
- `AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY=1`

When required, failures are reported through `readiness_gates`,
`readiness_failures`, status degradation, and attention flags. When observe-only,
failures remain visible without failing readiness. Choose the mode deliberately
for paper and live rollout.

## Operating Cadence

### Daily Pre-open

1. `curl -s http://127.0.0.1:9001/healthz | jq .`
2. Confirm:
   - `status="healthy"` or expected `market_closed`
   - no unexpected `attention_flags`
   - broker `positions_count` and `open_orders_count` are expected
3. `journalctl -u ai-trading.service -n 200 --no-pager`
4. Confirm:
   - no repeated provider/broker errors
   - no reconciliation freeze events
   - no unexpected ML / RL governance changes

### Daily Post-close

1. Compare paper results against offline replay for the same session.
2. Review:
   - trade count
   - hold times
   - exit reasons
   - slippage / reject signals
3. Confirm EOD flattening behavior if the mandate is flat-at-close.

### Weekly

1. Review runtime performance and go/no-go reports.
2. Review top reject reasons and stale pending counts.
3. Review any shadow/challenger learning outputs.
4. Refresh the list of known incident classes and confirm each still has coverage.

### Monthly

1. Run an incident review:
   - what failed
   - what was detected late
   - what lacked a runbook
2. Turn every real incident into:
   - one regression test
   - one operator note or alert improvement
   - one control-matrix update

## Escalation Triggers

Pause new openings and investigate immediately if any of the following occurs:

- unexpected non-flat positions before the paper/live open
- broker reconciliation mismatch burst or openings freeze
- provider safe-mode or halt flag activation
- repeated submit rejects or stale pending growth
- runtime unexpectedly leaving or re-entering no-ML mode
- health status degrades for reasons other than the known closed-market warm state

## What "Robust Enough" Means Here

The system is robust enough for the next phase when:

- pre-open checks are consistently clean
- replay and paper stay directionally aligned
- no unexplained non-flat / stale-order / reconciliation incidents recur
- learning promotions are blocked only by actual performance, not plumbing
- every meaningful failure mode in the control matrix has:
  - a guardrail
  - a signal
  - a drill
  - an operator action
