# Institutional 30/60/90 Roadmap

## Objective
Move the bot from feature-complete paper trading to institutional-grade operational discipline.

Institutional-grade in this repository means:
- deterministic execution and replay behavior
- fail-fast runtime safety contracts
- measurable execution quality (TCA) with bounded adaptation
- deploy gating that blocks unsafe releases
- documented and drilled incident response

## Day 0-30 (Stabilize and Enforce)
Focus: hard controls, observability, and deployment blocking criteria.

### Deliverables
1. Enforce mandatory CI gates for runtime safety and P0/P1 tests.
2. Enforce fail-fast order-type capability checks with explicit capability configuration.
3. Align environment key usage for pretrade controls and throttles.
4. Activate SLO threshold definitions and operational runbooks.
5. Establish canary-first deployment as the default release mode.

### Acceptance Criteria
1. `ci/scripts/institutional_gates.sh` passes in CI for every mainline merge.
2. No release can proceed if required P0/P1 tests fail.
3. No paper/live startup with ambiguous order-type capabilities.
4. No deploy without runbook references and SLO thresholds present.

### Verification Commands
```bash
bash ci/scripts/institutional_gates.sh
pytest -q tests/test_runtime_contract_no_stubs.py tests/test_order_type_startup_failfast.py
pytest -q tests/test_pretrade_env_key_alignment.py
```

## Day 31-60 (Measure and Govern)
Focus: execution-quality governance and replay-backed confidence.

### Deliverables
1. Daily TCA report review with top slippage/reject contributors.
2. Replay determinism check in CI and pre-deploy.
3. OMS lifecycle parity replay check in CI and pre-deploy.
4. Walk-forward + leakage guard outputs tracked per release candidate.
5. Model artifact verification policy enforced in deployment checklist.
6. Phase 2 execution-edge baseline includes calibration sufficiency diagnostics and conservative routing threshold recommendations.

### Acceptance Criteria
1. Daily execution report generated on schedule and archived.
2. Replay output hash remains stable for fixed seed/data window.
3. OMS lifecycle parity replay reports zero live/simulated stream mismatches.
4. Leakage checks fail hard when any horizon contamination is introduced.
5. Live promotions require verified model artifact manifests.
6. Adaptive routing remains disabled unless Phase 2 diagnostics are sufficient and release sign-off explicitly enables `AI_TRADING_PHASE2_EXECUTION_EDGE_ROUTING_ENABLED`.

### Verification Commands
```bash
pytest -q tests/test_tca_implementation_shortfall.py tests/test_execution_report_daily_rollup.py
pytest -q tests/test_replay_engine_deterministic.py tests/test_walk_forward_no_leakage.py
pytest -q tests/unit/test_oms_backtest_lifecycle_parity.py tests/unit/test_oms_lifecycle_parity_replay_tool.py
python3 -m ai_trading.tools.oms_lifecycle_parity_replay --fixture tests/data/oms_lifecycle_parity_fixture.json
python3 -m ai_trading.tools.update_phase2_execution_baseline --window-days 30
AI_TRADING_INSTITUTIONAL_REQUIRE_PHASE2_GATE=1 bash ci/scripts/institutional_gates.sh
pytest -q tests/test_model_artifacts.py tests/test_model_verification_policy.py
```

## Day 61-90 (Scale and Resilience)
Focus: portfolio governance, adaptation safety, and incident readiness.

### Deliverables
1. Quarantine/learning/allocation outcomes reviewed weekly with rollback criteria.
2. SLO breach alerts tied to action runbooks with owner/on-call assignment.
3. Disaster exercises for kill-switch, broker outage, and degraded data behavior.
4. Promotion review for moving from paper-only canary to broader symbol set.
5. Pre-open acceptance gate drilled with flat-start and strict OMS readiness settings.
6. Live KPI breach persistence drilled through dry-run and rollback-enabled modes.

### Acceptance Criteria
1. Quarantine and learning updates are bounded, logged, and reproducible.
2. On-call can execute each runbook without ad-hoc decisions.
3. Canary release can automatically rollback on threshold breach.
4. Deployment sign-off includes replay/TCA/leakage/model-governance results.
5. `/healthz` exposes required OMS readiness failures when strict OMS gates are enabled.
6. Pre-open operator drill blocks required flat-start or OMS failures before market entry.

### Verification Commands
```bash
pytest -q tests/test_quarantine_triggers_and_blocks.py tests/test_post_trade_learning_bounded_updates.py
pytest -q tests/test_allocation_weight_updates.py tests/test_portfolio_limits_vol_targeting.py
pytest -q tests/scripts/test_pre_open_acceptance_gate.py tests/main/test_runtime_governance_hooks.py
python3 scripts/pre_open_acceptance_gate.py --json
```

## Weekly Operating Cadence
1. Monday: review prior-week TCA, reject/cancel trends, and SLO breaches.
2. Wednesday: replay + walk-forward comparison for release candidate.
3. Friday: canary readiness review and go/no-go decision using acceptance matrix.

## Exit Criteria (Institutional Readiness Baseline)
1. `ci/scripts/institutional_gates.sh` green for 4 consecutive weekly releases.
2. No unresolved critical runbook gaps.
3. No unresolved high-severity leakage or model-verification defects.
4. Canary rollback workflow tested successfully at least once.
