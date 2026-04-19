# Trading System Checklist

Last reviewed: `2026-04-18`

This is the repo-specific version of "what good looks like" for this trading
system. It is organized into:

1. already implemented well
2. partially implemented / mixed
3. highest-priority gaps to fix next

The goal is not abstract architecture purity. The goal is to make the bot easy
to reason about under stress, especially around data quality, risk gates,
execution, persistence, and operator recovery.

## 1. Already Implemented Well

### Runtime contract and fail-fast startup

- Canonical config access is centralized through `ai_trading.config.management`.
- Startup/runtime invariants are documented in [AGENTS.md](../AGENTS.md) and
  [ARCHITECTURE.md](../ARCHITECTURE.md).
- Paper/live runtime contract checks and no-stub enforcement exist in:
  - [ai_trading/core/runtime_contract.py](../ai_trading/core/runtime_contract.py)
  - [ai_trading/core/bot_engine.py](../ai_trading/core/bot_engine.py)
- Acceptance coverage exists in:
  - [docs/acceptance_matrix.md](acceptance_matrix.md)

### Backtesting vs production-faithful replay separation

- Fast research path is clearly separated from production-faithful replay:
  - [ai_trading/strategies/backtester.py](../ai_trading/strategies/backtester.py)
  - [ai_trading/tools/offline_replay.py](../ai_trading/tools/offline_replay.py)
- This split is documented in:
  - [README.md](../README.md)
  - [docs/index.md](index.md)

### Health surfaces and operator visibility

- Canonical health payload construction is centralized in:
  - [ai_trading/health_payload.py](../ai_trading/health_payload.py)
- Main API and standalone health entrypoints share the same health semantics:
  - [ai_trading/app.py](../ai_trading/app.py)
  - [ai_trading/health.py](../ai_trading/health.py)
- Control-plane/operator surfaces exist and are useful:
  - [ai_trading/app.py](../ai_trading/app.py)
  - [ai_trading/services/control_plane.py](../ai_trading/services/control_plane.py)
- Broker counts and readiness details are exposed in health/control-plane output.

### Idempotency, reconciliation, and OMS persistence

- Stable order identity support exists:
  - [ai_trading/utils/ids.py](../ai_trading/utils/ids.py)
- Durable OMS / restart reconciliation infrastructure exists:
  - [docs/runbooks/restart_reconciliation.md](runbooks/restart_reconciliation.md)
  - [docs/persistence/sprint1_adr.md](persistence/sprint1_adr.md)
  - [ai_trading/services/reconciliation.py](../ai_trading/services/reconciliation.py)
- Runtime config explicitly supports OMS idempotency and decision records:
  - [ai_trading/config/runtime.py](../ai_trading/config/runtime.py)

### Model governance is more mature than typical small trading bots

- Promotion, rollback, approval, and governance surfaces exist:
  - [docs/model_governance.md](model_governance.md)
  - [ai_trading/training/after_hours.py](../ai_trading/training/after_hours.py)
  - [ai_trading/services/governance.py](../ai_trading/services/governance.py)
- Runtime model loading now fails closed instead of silently using placeholder artifacts.

### Robustness work already formalized

- A concrete operational robustness matrix already exists:
  - [docs/ROBUSTNESS_AUDIT.md](ROBUSTNESS_AUDIT.md)
- This repo already treats:
  - stale/degraded data
  - reconciliation drift
  - broker health
  - pre-open readiness
  - model governance
  as first-class operational concerns.

## 2. Partially Implemented / Mixed

### Separation of concerns exists, but `bot_engine.py` is still the gravity well

- The system has real modules for data, risk, services, governance, replay, and health.
- The netting-cycle decision path now builds canonical decision records through:
  - [ai_trading/core/netting.py](../ai_trading/core/netting.py)
  instead of having `bot_engine.py` assemble that contract state itself.
- Pre-submit execution intent context is now built through:
  - [ai_trading/core/execution_intent.py](../ai_trading/core/execution_intent.py)
  rather than being assembled inline in the live netting loop.
- Approval and pre-submit guard helpers now live in:
  - [ai_trading/core/execution_guards.py](../ai_trading/core/execution_guards.py)
  instead of being fully expanded inline in `bot_engine.py`.
- Post-submit outcome normalization and TCA/result assembly now live in:
  - [ai_trading/core/execution_outcome.py](../ai_trading/core/execution_outcome.py)
  instead of being fully expanded inline in `bot_engine.py`.
- Canonical decision record building/counting/observation capture now lives in:
  - [ai_trading/core/decision_log.py](../ai_trading/core/decision_log.py)
  instead of being owned as a large local block inside the netting loop.
- The top-level `symbol="ALL"` halt/block decision records now also flow through
  that shared recorder instead of repeating zero-target record assembly inline.
- Sleeve/universe/position preparation for the netting loop now lives in:
  - [ai_trading/core/netting_cycle_setup.py](../ai_trading/core/netting_cycle_setup.py)
  instead of being fully expanded inline in `bot_engine.py`.
- Replay-quality loading and policy-toggle override setup for ranking now lives in:
  - [ai_trading/core/netting_rank_prelude.py](../ai_trading/core/netting_rank_prelude.py)
  instead of being fully expanded inline in `bot_engine.py`.
- Learned ranking-state assembly for bandit, realized-edge, expected-capture,
  execution-learning, rejection-concentration, and portfolio-log-growth inputs
  now lives in:
  - [ai_trading/core/netting_learning_state.py](../ai_trading/core/netting_learning_state.py)
  instead of being fully expanded inline in `bot_engine.py`.
- Per-symbol candidate ranking and opportunity-quality scoring now live in:
  - [ai_trading/core/netting_candidate_rank.py](../ai_trading/core/netting_candidate_rank.py)
  instead of keeping the entire scoring core inline in `bot_engine.py`.
- Global execution-control context for SLO derisk, rollout/capital ramp,
  capacity throttle, primary-feed derisk, gate auto-disable, and uncertainty
  capital setup now lives in:
  - [ai_trading/core/netting_execution_context.py](../ai_trading/core/netting_execution_context.py)
  instead of being fully expanded inline in `bot_engine.py`.
- Per-symbol quantity adjustment controls for adaptive sizing, uncertainty
  capital scaling, reversal clamps, and symbol/min-notional caps now live in:
  - [ai_trading/core/netting_symbol_adjustments.py](../ai_trading/core/netting_symbol_adjustments.py)
  instead of keeping that full mutation block inline in `bot_engine.py`.
- Pre-submit broker gate orchestration for portfolio-optimizer vetoes,
  auth cooldown, breaker/ledger dedupe, pretrade validation, quote basis,
  and NBBO opening checks now lives in:
  - [ai_trading/core/netting_submit_prelude.py](../ai_trading/core/netting_submit_prelude.py)
  instead of keeping that whole broker-prelude block inline in `bot_engine.py`.
- Broker submit outcome handling for exception/`None`/success paths now lives in:
  - [ai_trading/core/netting_submit_execution.py](../ai_trading/core/netting_submit_execution.py)
  instead of keeping the whole submit-result branch inline in `bot_engine.py`.
- Cycle-level analytics flush, reject-summary logging, SLO logging, and
  quarantine persistence now live in:
  - [ai_trading/core/netting_cycle_summary.py](../ai_trading/core/netting_cycle_summary.py)
  instead of keeping that full epilogue inline in `bot_engine.py`.
- Early symbol-level prelude gating for policy-ablation sleeve blocks, signal
  age/time-stop handling, opportunity/staleness checks, and capital-ramp
  scaling now lives in:
  - [ai_trading/core/netting_symbol_prelude.py](../ai_trading/core/netting_symbol_prelude.py)
  instead of keeping that whole early pre-submit block inline in `bot_engine.py`.
- Pre-approval symbol orchestration for sell-qty clipping, opening-trade
  prechecks, cost-aware edge gating, approval evaluation, and approval-driven
  delta/side adjustment now lives in:
  - [ai_trading/core/netting_symbol_approval.py](../ai_trading/core/netting_symbol_approval.py)
  instead of keeping that full pre-submit approval block inline in `bot_engine.py`.
- The remaining per-symbol `_run_netting_cycle` glue now flows through:
  - [ai_trading/core/netting_symbol_cycle.py](../ai_trading/core/netting_symbol_cycle.py)
  so the hot loop in `bot_engine.py` is mostly setup, dispatch, and summary
  instead of owning the full symbol-level state machine inline.
- `run_all_trades_worker` cycle bootstrap for policy/config preflight, market/API
  gating, startup cleanup, safe-mode cancellation, cache reset, and cycle
  activation now lives in:
  - [ai_trading/core/run_all_trades_prelude.py](../ai_trading/core/run_all_trades_prelude.py)
  instead of keeping that full pre-cycle block inline in `bot_engine.py`.
- But [ai_trading/core/bot_engine.py](../ai_trading/core/bot_engine.py) still owns too much:
  - strategy orchestration
  - risk gating
  - execution coordination
  - provider fallback behavior
  - model/runtime behavior
  - parts of journaling and operator behavior

Verdict:
- `Partially implemented`
- This is the largest architectural concentration risk left.

### Canonical contracts are stronger, but not fully standardized

- Some boundaries are explicit and typed.
- Canonical decisioning contracts now exist in:
  - [ai_trading/contracts/decisioning.py](../ai_trading/contracts/decisioning.py)
  - [ai_trading/contracts/market.py](../ai_trading/contracts/market.py)
  - [ai_trading/strategies/base.py](../ai_trading/strategies/base.py)
  - [ai_trading/oms/pretrade.py](../ai_trading/oms/pretrade.py)
- But the system still has a lot of loose `dict[str, Any]` payload flow,
  especially in orchestration, health, runtime state, and policy metadata.
- Request models are now more stable again in:
  - [ai_trading/data/models.py](../ai_trading/data/models.py)
- The contract layer now covers:
  - `Bar`
  - `Quote`
  - `Signal`
  - `RiskDecision`
  - `OrderIntent`
  - `ExecutionResult`
  - `PositionSnapshot`
  - `BrokerOrderSnapshot`
- But those contracts are still not used uniformly across every runtime path.

Verdict:
- `Partially implemented`

### Decision journaling exists in pieces, but not yet as one canonical explainability record

- Config already supports decision record JSONL:
  - [ai_trading/config/runtime.py](../ai_trading/config/runtime.py)
- Decision records now emit a canonical `decision_journal` envelope with typed:
  - `Signal`
  - `RiskDecision`
  - `OrderIntent`
  - `ExecutionResult`
  - `BrokerOrderSnapshot`
  via [ai_trading/core/netting.py](../ai_trading/core/netting.py)
- Runtime reports, health payloads, and OMS ledgers provide useful evidence.
- The live netting cycle now creates those canonical contracts at record-build time,
  not only later during journal serialization.
- The canonical journal now carries stable fields for:
  - `event`
  - `provider`
  - `feed`
  - `target_delta_shares`
  - `client_order_id`
  - `broker_result`
  - `reasons`
- But there is still not one obviously canonical "per symbol / per cycle / per decision"
  journal that consistently captures:
  - data freshness
  - signal
  - risk outcome
  - target delta
  - intent
  - submission result
  - broker response

Verdict:
- `Partially implemented`

### Fallback behavior is much cleaner now, but not fully policy-pure

- The large fake runtime stub surfaces have been cleaned up.
- But the repo still intentionally supports explicit degraded/fallback behavior in areas like:
  - provider switching
  - market calendar fallback tables
  - retry wrappers without Tenacity
  - sentiment unavailable => neutral `available=False`
- Those are visible and more honest now, but they still need clearer policy boundaries:
  "allowed degradation" vs "must fail closed."

Verdict:
- `Partially implemented`

### Reconciliation is strong, but not yet maximally enforced

- Reconciliation logic, controls, and runbooks are present.
- Health/control-plane visibility is present.
- But [docs/ROBUSTNESS_AUDIT.md](ROBUSTNESS_AUDIT.md) already notes that some OMS
  invariant/parity checks are still optional rather than hard readiness gates.

Verdict:
- `Partially implemented`

### Replay/live parity is now a named gate, but not yet the only operational standard

- A shared replay/live parity gate summary now exists in:
  - [ai_trading/governance/replay_live_parity.py](../ai_trading/governance/replay_live_parity.py)
- That gate is surfaced in:
  - [ai_trading/health_payload.py](../ai_trading/health_payload.py)
  - [ai_trading/tools/runtime_performance_report.py](../ai_trading/tools/runtime_performance_report.py)
- It now combines replay governance freshness/counterfactual status with OMS
  lifecycle parity into one operator-facing pass/fail object.
- But parity is still not yet the sole canonical pre-rollout / pre-promotion
  contract across every workflow in the repo.

Verdict:
- `Mostly implemented`

## 3. Highest-Priority Gaps To Fix Next

These are the next upgrades that would most improve production robustness.

### 1. Thin `bot_engine.py` by responsibility, not by file count

Target:
- move business logic out of [ai_trading/core/bot_engine.py](../ai_trading/core/bot_engine.py)
  into clearer modules such as:
  - market-data evaluation
  - signal-to-intent translation
  - final OMS risk gate
  - execution result handling
  - reconciliation scheduling

Why it matters:
- this is the biggest remaining "hard to reason about under stress" risk

### 2. Formalize canonical domain contracts

Target:
- standardize internal contracts for:
  - `Bar`
  - `Quote`
  - `Signal`
  - `RiskDecision`
  - `OrderIntent`
  - `ExecutionResult`
  - `PositionSnapshot`
  - `BrokerOrderSnapshot`

Why it matters:
- it reduces ambiguous dict payloads and makes replay/live parity easier to verify

### 3. Make the decision journal fully canonical

Target:
- one append-only decision record per evaluated symbol/bar/cycle with stable fields:
  - `event`
  - `symbol`
  - `bar_ts`
  - `provider`
  - `feed`
  - `signal`
  - `risk_decision`
  - `target_delta`
  - `client_order_id`
  - `submitted`
  - `broker_result`
  - `reasons`

Why it matters:
- this is the fastest path to explainability during real incidents

### 4. Promote replay/live parity to a first-class gate

Target:
- make "paper/live behavior should be explainably close to offline replay on the same data"
  a named operational standard, not just a good habit

Why it matters:
- this repo already has strong replay infrastructure, and it is one of the most
  valuable controls available

Use:
- [ai_trading/tools/offline_replay.py](../ai_trading/tools/offline_replay.py)
- replay-related runbooks and governance reports
- [ai_trading/governance/replay_live_parity.py](../ai_trading/governance/replay_live_parity.py)

### 5. Tighten the final OMS boundary

Target:
- ensure the final execution boundary always owns:
  - duplicate prevention
  - quote freshness
  - market-hours restriction
  - spread / price sanity
  - symbol/gross/net exposure
  - kill switch
  - broker readiness

Why it matters:
- upstream correctness is never enough in a live trading system

### 6. Decide which remaining degraded paths should fail closed

Most important policy decisions left:
- should [ai_trading/analysis/sentiment.py](../ai_trading/analysis/sentiment.py)
  continue returning explicit neutral fallback, or should some modes hard-fail?
- should execution-engine stub recovery paths in
  [ai_trading/core/bot_engine.py](../ai_trading/core/bot_engine.py) be removed
  outside test-only environments?
- should OMS invariant / lifecycle parity checks become hard health gates by default?

Why it matters:
- these are not code-quality questions anymore; they are production-policy questions

## 4. Recommended Next Order

If the goal is "most robustness gain for the next engineering effort," the order
I would use is:

1. define canonical `Signal`, `RiskDecision`, and `OrderIntent` contracts
2. build one canonical decision journal around those contracts
3. split the most overloaded parts of `bot_engine.py` along those boundaries
4. tighten replay/live parity checks into a named operational gate
5. decide which remaining degraded-mode behaviors should become fail-closed

## 5. Bottom Line

This repo is already ahead of many trading bots in:

- health and operator visibility
- replay tooling
- reconciliation thinking
- governance and promotion controls
- runtime contract enforcement

The biggest remaining risks are not "missing a few helper functions."
They are:

- oversized orchestration gravity in `bot_engine.py`
- incomplete canonical domain contracts
- incomplete decision journaling
- unresolved policy choices around what may degrade vs what must stop

That is a good place to be. The system no longer looks underbuilt.
It now mostly needs sharper boundaries and sharper operational contracts.
