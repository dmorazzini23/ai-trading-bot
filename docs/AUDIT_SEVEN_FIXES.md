# Seven audit findings: corrections

September 11, 2026. User authorized all seven fixes end-to-end. This patch changes
execution/reconciliation and governance correctness; no strategy search, holdout
evaluation, promotion, order or notification is authorized by it.

1. Position-target reconciliation validates the complete response before updating
   snapshots or pruning targets. Missing, malformed, nonfinite and duplicate
   position rows fail reconciliation and preserve existing protection. Valid
   zero quantities remain flat-position evidence.
2. Opposite-order cancellation counts only verified canceled/expired/rejected
   orders. Errors, timeouts, full fills and reported partial fills block the new
   intent. Exposure changes require a new decision cycle; cancellation success
   is never logged solely because a cancel request returned.
3. Broker read retry and reconciliation explicitly catch Alpaca SDK APIError.
   HTTP status remains available for classification. 429 and 5xx are retryable;
   authentication failures halt without retrying. This does not add retries to
   non-idempotent order submission.
4. OMS reconciliation rejects nonfinite internal/broker quantities and invalid
   tolerances. Broker parsing rejects missing and nonfinite quantity evidence.
5. The OOF promotion confirmation check requires an enabled, explicitly passing
   check; disabled/not-required checks cannot supply confirmation. Other model
   eligibility requirements remain in force.
6. Approval-journal writes are locked, flushed and fsynced. Failed persistence
   raises through the service into HTTP 503; no success audit is emitted before
   the primary approval/rejection write succeeds. Other journal callers preserve
   their existing failure contract.
7. Shadow updates require immutable session_id, timezone-aware source_start and
   source_end. Source intervals must be ordered, nonoverlapping and not future
   dated. A per-model lock covers reading, deduplication and atomic metrics write.
   Identical retries are no-ops, changed payloads under the same identity fail,
   and last_updated is the observation end rather than processing time.

## Shadow evidence compatibility

Existing accumulated metrics without observation identities cannot accept new
updates as if their provenance were known. They require explicit evidence review;
the patch does not reset them, fabricate identities or silently discard history.
New sessions must supply the required fields. There was no production call site
for this update API in the audited tree; tests now use real distinct intervals
instead of counting duplicate payloads as consecutive sessions.

## Regression and runtime checks

New coverage is in tests/test_audit_safety_regressions.py and the operator API
tests. It covers mixed-invalid snapshots, NaN/Infinity, invalid tolerances, real
SDK 429/503/401 exceptions, reconciliation halt state, cancellation failure and
partial fills, disabled confirmation, unsuccessful rejection persistence,
source-time freshness, identity conflicts and concurrent duplicate retries.

The existing execution/runtime-control, after-hours training, institutional
promotion, governance and service suites are checked alongside those regressions.
See docs/CODEX_HANDOFF.md for final command results and deployment evidence.

## Risk and rollback

These failures now stop the affected operation instead of treating unavailable
evidence as success. This may expose malformed upstream data or callers missing
session identity. Keep those failures visible; do not bypass them to generate
trades. Roll back only this patch if it introduces a regression, preserving the
research-reset settings, consumed ledgers and unrelated working-tree changes.
