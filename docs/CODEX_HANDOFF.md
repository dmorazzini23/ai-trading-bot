# Current handoff

Updated September 11, 2026, 14:54 UTC. User authorized all three: rollback-counter
repair, existing replay diagnosis, and governed model replacement decision.
Report: docs/ROLLBACK_REPLAY_MODEL_DECISION.md.

## Current implementation

- ai_trading/governance/promotion.py serializes KPI evaluation/accounting/actions
  under a shared file lock; atomic state writes fail closed. Observation identities
  prevent poll/retry inflation across restarts and healthy resets. Unverified
  legacy counts are diagnostic only. Conflicting payload identity is rejected.
- ai_trading/main.py uses persisted counts without increasing a local streak;
  source observation timestamps supply runtime identities.
- ai_trading/monitoring/slo.py exposes last_observation_at.
- Regression coverage: tests/governance/test_kpi_observation_accounting.py,
  tests/main/test_runtime_governance_hooks.py and updated institutional/governance
  fixtures use distinct evidence where a new observation is intended.

## Validation

- bash scripts/agent_validate_changed.sh --market-hours --skip-runtime-smoke:
  lint, mypy (49 sources), compile and 720 selected tests passed.
  /tmp/three-items-validation.log.
- Final targeted rollback/scheduler tests: 14 passed before one additional
  distinct-concurrent-writers test was included in the selected validation.
  /tmp/rollback-targeted-final.log.
- Final lint/compile for updated terminal-status guard and concurrency test passed.
- Non-sending incident snapshot passed. No messages or orders sent by this task.
- Saved replay hash, sample counts, gross-minus-drag identity and attribution
  residual verified directly; no new backtest, experiment or source replay run.

## Evidence decisions

- Latest existing replay (September 11 09:23 UTC): candidate -18.281386 bps,
  3174 next-observation markouts. Gross -3.941979 minus execution-price drag
  14.339406 bps. Baseline also negative (-17.822979); cap contrast -0.458407 bps.
  This is simulated markout accounting, not live P&L or pure fee attribution.
- Source report overlaps protected holdout; do not tune/select from its subgroups.
- Replacement decision: retain abstention. Selected model ~55.8 days old against
  14-day limit and newest of 187 ml_edge registered entries (July 17).
- Latest training report: no_qualified_candidate; logreg post-cost OOF -19.025753
  bps, 0/5 profitable folds, no runtime/promotion/live-money authority.
- Research reset remains active, no gate weakened and no replacement installed.

## Deployment

- Preflight 14:53:38 UTC: paper, broker connected/fresh, one position,
  zero open orders. /tmp/three-items-predeploy.json.
- Service restart succeeded. Active service and fresh broker at 14:54:46 UTC:
  one position, zero open orders, no broker failures. Only existing stale-model
  and replay flags remain. /tmp/three-items-health-final.json.
- Documentation-only validation and git diff --check passed.

## Earlier completed work and constraints

Seven audit fixes: docs/AUDIT_SEVEN_FIXES.md. Shadow-start/provenance follow-up:
docs/GOVERNANCE_FOLLOWUP_FIXES.md. Environment/reset fixes: docs/ENV_RESET_FIXES.md.
All were previously validated/deployed. Preserve these and unrelated dirty
research/efficiency changes. No commit/reset to tidy the tree.
September 9–December 8 holdout and consumed campaign budgets remain. October 8
does not auto-release research. Historical holdout isolation is not established.
Legacy shadow evidence requires review; explicit administrative force promotion
still bypasses ordinary eligibility. KPI source IDs establish deduplication,
not statistical independence; no-ID callers conservatively hash monitored values.
