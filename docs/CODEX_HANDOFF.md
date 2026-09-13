# Current handoff

## Timing contract implementation — September 12, 2026

See docs/REPLAY_TIMING_CONTRACT.md. Replay summary now persists explicit metric
anchors and exact 300-second observation diagnostics for every fill, including
excluded fills. Diagnostics have no qualification authority or alternate returns.
Existing metrics/gates, costs, model selection and research restrictions unchanged.
Isolated replay /tmp/timing-check/output/replay_hash_20260912.json matched prior
production source hash and every existing summary field. Correctly blocked exit 2.
Candidate exact observations missing: 33/83 decision-anchor, 43/83 fill-anchor.
Baseline missing: 201/539 decision-anchor, 249/539 fill-anchor. Counts reconcile.
32 focused tests passed; final exact-observation assertions: 2 passed plus lint.
Changed-file validator: 836 tests passed, lint/types (9 sources), compile;
/tmp/timing-validation.log. Used --market-hours --skip-runtime-smoke with separate
isolated replay, live health and non-sending checks. Docs-only and diff checks passed.
Health broker fresh, zero orders/positions;
non-sending incident snapshot passed. No service restart needed: scheduled CLI
loads checkout each invocation. Actual timer check remains pending September 14.

## Four-item evidence audit — September 12, 2026

See docs/REPLAY_BLOCKER_SCORECARD_20260912.md for source hashes and completion criteria.
Executed installed replay CLI with production paths/settings; artifact refreshed
17:46 UTC, correctly blocked on qualification. Manual systemd start requires an
unavailable password; actual timer adoption pending September 14 09:22:18 UTC.
Timing: 33/77 candidate markouts exceed five minutes; 47/77 fills delayed beyond
five minutes. Model metadata still has irregular elapsed label horizons.
Fresh broker reconciliation: 535 activities, 407 quantity matches, 407 unknown
fee totals; 83 fee records have no execution references. External evidence gap.
No tuning, training, order submission or gate changes. Documentation only.
12 focused replay-tool/accounting tests passed; artifact arithmetic assertions and
non-sending snapshot passed. Health 17:47 UTC broker fresh, zero orders/positions,
existing stale-model/replay flags. Next task: explicit governed timing contract
and coverage diagnostics; preserve holdout and consumed research budgets.

## Latest IOC implementation — September 12, 2026

See docs/REPLAY_IOC_FIXES.md. Simulator IOC uses a submission-time observation,
cancels any remainder and never waits for a later quote. Replay applies immediate
fills to exposure before the next cap check; summaries persist cancellations.
Qualification/cost/model/research gates unchanged. Earlier IOC blocker below is
resolved: isolated replay now writes an artifact and correctly fails qualification.

- Focused regressions: 51 passed; /tmp/replay-ioc-regressions.log.
- Isolated artifact: /tmp/replay-ioc-check/output/replay_hash_20260912.json.
  Candidate 77 usable + 6 excluded = 83 fills; baseline 525 + 14 = 539.
  All bounded markouts fill before expiry; three baseline IOC fills have zero delay.
  Candidate -9.750137 bps; positive edge, sample minimum and non-regression fail.
- Non-sending incident snapshot passed. Preflight 17:40 UTC: paper broker fresh,
  zero orders/positions; existing stale-model/replay attention flags only.
- Changed-file validator passed: 834 selected tests, lint, types (8 sources),
  compile; /tmp/replay-ioc-validation.log. Used --market-hours --skip-runtime-smoke
  with separate live health, incident snapshot and isolated replay checks.
- Paper service restart succeeded; service active and broker fresh/connected at
  17:43:03 UTC, zero positions/orders and no broker failures. Only existing
  stale-model/replay flags remain; /tmp/replay-ioc-health-final.json.
- Docs-only validator and git diff --check passed. Next: address named evidence
  gaps under research-reset governance; no qualification or promotion justified.

## Latest expiry implementation — September 12, 2026

See docs/REPLAY_EXPIRY_DIAGNOSTICS.md. Day/GTC and explicit expiry are propagated
through replay; governance missing TIF uses labeled day assumption. Day expiry
uses canonical NYSE close, precedes fills and preserves partial filled quantity.
Expired events and per-fill markout exclusion reasons persist in summaries.
No qualification, cost, horizon, model or research-setting changes.

- Changed-file validator: 827 selected tests passed, lint/types (8 sources), compile.
  /tmp/replay-expiry-validation.log.
- Focused regressions: 44 passed. /tmp/replay-expiry-regressions-final.log.
- Final simulator lint/compile and git diff --check passed.
- Initial isolated-run approval hit a usage limit. After its reset and user
  continuation, the normal approval path succeeded; no workaround used.
- Isolated replay then failed on unsupported recorded time_in_force=ioc before
  writing an artifact. /tmp/replay-expiry-check.log and
  /tmp/replay-expiry-check/summary.json. Do not report real-data expiry reconciliation
  as verified. IOC requires evidence-backed handling; no approximation introduced.
- Non-sending incident snapshot passed. Preflight 17:32 UTC: paper, fresh broker,
  zero positions/orders; existing stale-model/replay flags only.
- Service restart succeeded; active service and fresh broker at 17:34:15 UTC,
  zero positions/orders and no broker failures. Existing stale-model/replay flags
  remain. /tmp/replay-expiry-health-final.json.
- Documentation validation and final git diff --check passed.

## Latest evidence review — September 12, 2026

Completed four-item post-fix review: docs/POSTFIX_REPLAY_EVIDENCE_REVIEW.md.
No new scheduled artifact existed; systemd start required a sudo password.
Ran the same CLI/settings as aiuser with isolated /tmp/postfix-replay-check outputs.
Production artifact/service untouched. CLI wrote evidence then correctly blocked
on REPLAY_POLICY_NON_REGRESSION_FAILED.
68/68 candidate markouts passed field/timestamp/assumption/arithmetic checks.
7,396 shadow records rejected for missing/unsupported types; final 1,859 source
rows reduce to 911 by configured symbols. Candidate -10.888324 bps, 68 samples;
positive edge, sample minimum and edge non-regression fail.
Median fill delay 1 hour; median markout interval 10 minutes; maximum fill delay
94.75 hours. Selected model targets one 5Min bar but recorded training label
elapsed times are also irregular. Diagnostic-only, not qualification evidence.
Next gap: order expiry/time-in-force and explicit markout exclusion diagnostics.
No tuning, model fitting, live orders or promotion. Holdout restrictions remain.

## Latest implementation — September 12, 2026

Replay contract fixes implemented; see docs/REPLAY_CONTRACT_FIXES.md.
Recorder/journal and replay separate submission_status from executable type.
Unsupported types fail before simulator order creation; unknown opportunity intent
is rejected diagnostically, never guessed. Explicit historical order_type remains
accepted. Saved markouts include timestamps, intervals and simulator assumptions.
Qualification gates, model selection, costs and research restrictions unchanged.

Changed runtime: contracts/decisioning.py, core/decision_log.py,
core/bot_engine.py, execution/simulated_broker.py, replay/event_loop.py.
Regressions: tests/test_decision_recorder.py,
tests/test_replay_governance_async_parity.py,
tests/unit/test_simulated_broker_reproducible.py.

- Final focused synthetic replay/broker tests: 40 passed.
  /tmp/replay-contract-final-synthetic.log.
- Final recorder suite: 11 passed. /tmp/replay-recorder-final.log.
- Changed-file validation passed: lint/types (8 sources), compile, 823 tests.
  /tmp/replay-contract-validation-final.log.
- Final explicit-type fallback lint/compile passed after its last edit.
- Non-sending incident snapshot passed; git diff --check passed.
- Preflight 02:05 UTC: paper, fresh broker, zero positions/orders. Existing
  provider-backup/degraded and stale-model/replay flags remain.
  /tmp/replay-contract-preflight.json.
- Restart succeeded; active service and fresh broker at 02:10:05 UTC confirmed
  zero positions/orders and no broker failures. Only existing stale-model/replay
  flags remain. /tmp/replay-contract-health-final.json.
- Documentation validation and git diff --check passed.
- No new research replay, historical artifact rewrite, model or order initiated.

## Latest bounded audit — September 11, 2026, 14:58 UTC

Completed execution-price-drag audit: docs/REPLAY_EXECUTION_DRAG_AUDIT.md.
Arithmetic reconciles, but 2512/3174 markouts use not_submitted as an executable
order type (market-like behavior), simulator defaults supply spread/volatility,
and persisted rows lack timestamps needed to validate actual holding intervals.
Synthetic limit-vs-not_submitted reproduction confirmed the semantic difference.
No runtime code/config/model changes or new replay. Next corrective scope is in
the audit; do not infer historical order intent or tune from overlapping holdout.

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
