# Accounting and candidate follow-through

Verified September 7, 2026. This completes the implementation and current-data
checks for broker fee coverage, zero-sample diagnosis, and scheduled reporting.

## Broker accounting

The new `ai_trading.tools.broker_accounting_evidence` CLI reads the paper account
and paginates account activities using the pinned Alpaca client. The daily
workflow runs it before paper evidence review. Pagination truncation remains
explicitly incomplete. Duplicate/conflicting activities and rejected local
records are counted. No orders or alerts are sent.

The actual 90-day fetch returned 548 activities: 465 fills and 83 fees.
Broker order IDs recovered 418 historical order-account links, and all 418
order quantities matched the local records. These links do not prove individual
fill identity or fee completeness, and the original records are unchanged.
Another 6,966 local rows remain outside the verified account links.

All 418 linked records lack verified total fees. The returned fee activities
include aggregate charges, such as CAT fees, with no individual fill or order
ID. The report preserves their subtype, currency, signed amount, date and
unallocated status. It does not distribute aggregate fees across executions or
infer zero fees when no charge appears. This follows the distinction between
trade and non-trade records in the [Alpaca account activity contract](https://docs.alpaca.markets/us/docs/account-activities).

## Candidate decision

`candidate_abstention_review` traces development-only threshold diagnostics into
each accelerator candidate and the operator report. It does not use holdout
outcomes or change thresholds.

The inspected one-bar hist-gradient risk-adjusted candidate failed all 20
threshold trials across five inner-validation folds. Eight trials had
nonpositive gross edge; 12 had positive gross edge smaller than the assumed
6 bps round-trip costs. Even the best tested development net result was
-4.3331 bps. The causal chain is threshold rejection, abstention thresholds,
zero selected samples, regime abstention, and registry rejection.

The candidate is retired on this evidence. No bounded threshold correction is
justified. The existing experiment ledger independently rejects a repeat using
`evidence_already_evaluated`; its broader hypothesis remains subject to the
existing stopping rules for new evidence. No additional training or untouched
holdout consumption was necessary for this diagnosis.

## Scheduled reporting and artifacts

The daily plan was generated with the installed runtime environment. It requires
the current acquisition manifest and validated data, schedules accounting before
paper review, and passes the current training, selection, replay and accounting
reports into that review. The operator summary now includes accounting counts,
candidate retirement diagnostics and incomplete-accounting reasons.

The new accounting and combined-review steps were run against actual inputs.
Tonight's full scheduled run has not yet occurred; its timer is already active.
Research jobs load the changed modules in fresh processes, so no further live
service restart is needed for this task.

Artifacts in `artifacts/accounting_followthrough/` include:

- `account_activities.json` and `accounting_review.json`: fetched source and
  reconciliation results.
- `candidate_abstention.json` and `reviewed_training.json`: development diagnosis
  and a derived copy of the earlier training report with that diagnosis attached.
- `scheduled_plan/`: the runtime-environment daily plan.
- `paper_evidence_review.json`: the combined actual-evidence review.
- `operator_summary_verification.json`: verified propagation of all 418 matched
  order quantities and the retired-candidate diagnosis into the operator view.
- `health.json`: live readiness smoke check; the stale-model gate remains active.

## Validation and remaining dependencies

Regression coverage includes pagination exhaustion, accounting quantity joins,
legacy order-account recovery without source mutation, contradictory identities,
unknown fees, development-only retirement and scheduled step ordering. The
changed-file validation and final targeted results are recorded in
`/tmp/accounting-validation.log` and `/tmp/accounting-final-targeted.log`.
`bash scripts/agent_validate_changed.sh --skip-runtime-smoke` passed 728 tests,
lint, mypy and compilation. The final targeted suite passed 41 tests, and the
final accounting module and test passed lint and mypy after order-link recovery.
The non-sending incident snapshot passed. Health remains HTTP 503 because the
required serving model is stale; no trading gate was relaxed.

Exact per-fill total fees and a complete observed paper session remain missing
evidence. Broker aggregate charges alone cannot resolve those requirements.
Rollback consists of reverting the new accounting/abstention modules and their
training/review/scheduler wiring, preserving the preceding fee-contract fixes.
