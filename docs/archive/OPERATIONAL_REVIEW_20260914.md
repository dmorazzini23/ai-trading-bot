# Operational evidence review — September 14, 2026

Reviewed at approximately 02:54–02:58 UTC. This is a current diagnostic snapshot,
not a completed September 15–21 weekly review or post-deployment regular-session
qualification. No runtime code, configuration, model, gate or schedule changed.

## Live operation

The service is healthy/ready and the broker snapshot is fresh and connected,
with zero positions and open orders. Existing flags remain
`replay_live_parity_gate_failed` and `required_model_stale`. Bounded journal
inspection since the 02:30 UTC deployment found no matching Traceback/ERROR or
input-provenance/contract records. Absence of those records is not successful
inference evidence. Health snapshot: `/tmp/four-priorities-health.json`.

The canonical calendar places today's session at 13:30–20:00 UTC. It has not
started. Inspect actual batch provenance, feature rejection reasons and timestamp
alignment during that session; retain any stale-model abstention as a limitation.
Do not force inference, trades or model substitution to complete this check.

## Scheduled evidence and training pause

The canonical `verify_run` and `verify_operator` checks verified the September 11
daily, September 12 weekly and September 13 Sunday reports and their artifacts.
Each ran only broker accounting, paper evidence review and reset scorecard steps;
all three steps passed. These are workflow results, not strategy qualification.
The current `training_block_reason()` returns `research_reset_active`.
Reset-policy and scorecard regression suites passed: 22 tests.

Observed next timer triggers: daily September 14 20:39 UTC; weekly September 19
14:18 UTC. Times include timer randomization and should be refreshed at review.
The separate user evidence-verification timer has no next trigger: it was a
September 7–8 one-shot. It does not cover today's session. Recurring research
timers remain active; no new automatic follow-up was installed by this review.

## Fee evidence

Fresh read-only paper snapshot: 535 activities, complete pagination, 407 matched
local order quantities and 407 unknown per-fill fee totals. Raw activity schema:
452 fill rows with no fee amount, 83 fee rows with no execution reference.
Raw activity rows and matched local orders are different populations.
Private capture: `/tmp/four-priorities-activities.json`; reconciliation:
`/tmp/four-priorities-accounting.json`.

Alpaca documents statement/confirmation retrieval through its Documents API:
https://docs.alpaca.markets/us/docs/statements-and-confirms
The configured paper Trading client does not provide that Broker API document
integration. No authenticated statement/export or broker completeness confirmation
was obtained. Documentation alone does not establish availability for this paper
account or prove that a document contains complete execution-level fees.

Resolution requires broker-authoritative execution/account identifiers, currency,
quantity, timestamp and total charges with an explicit completeness basis. Missing
charges remain unknown; account-level charges must not be allocated by guesswork.
No broker contact, notification, credential change or order was made.

## Current scorecard and next review

Generated `artifacts/research_reset/operational_review_20260914_scorecard.json`
using the canonical scorecard CLI and freshly reconciled paper/accounting reports.
Window: September 7 02:56 UTC through September 14 02:56 UTC.

| Evidence area | Current finding |
| --- | --- |
| Operational correctness | Service recovered; regular-session behavior pending |
| Data/sample completeness | Sampling support remains insufficient |
| Execution evidence | 12 unique observed fills; 0 complete causal chains; all 12 missing verified fees and causal quotes |
| Research accounting | 2 concluded trials in trailing seven days; no trial launched by this review |
| Strategy qualification | Untouched performance unavailable; holdout not evaluated |

The complete-chain fraction is 0/12 for the observed seven-day population, not a
broker-completeness claim or a strategy return. Paper-session reconciliation and
net execution comparison remain unsupported. September 15–21 has not begun;
review the September 19 weekly output after it runs and compare these same
evidence areas. Preserve consumed budgets and the September 9–December 8 holdout.

Validation artifacts: `/tmp/four-priorities-tests.log` (22 passed),
`/tmp/four-priorities-refresh.log`, `/tmp/four-priorities-paper.log`, and
`/tmp/four-priorities-scorecard.log` (all three CLIs exited zero while retaining
evidence-pending status). No additional runtime patch required deployment.
