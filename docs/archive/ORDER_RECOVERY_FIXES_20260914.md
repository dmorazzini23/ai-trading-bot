# Order recovery fixes — September 14, 2026

The four findings in ORDER_RECOVERY_AUDIT_20260914.md are corrected in source.

## Changes

- IntentStore uses conditional database transitions: delayed submission errors
  only affect pending/submitting intents; acknowledgements preserve terminal and
  partial-fill states; fills preserve terminal states; later terminal callbacks
  cannot overwrite an already terminal state. Late fills still enter the ledger.
- External cumulative fill callbacks are handled by IntentStore in one database
  transaction. An intent-row write lock precedes the persisted quantity read and
  delta insert. Duplicate/decreasing totals insert nothing. Independent store
  connections and process restarts use durable totals, not a manager cache.
  Invalid nonfinite/negative quantities fail before persistence. No schema change.
- The optional position reconciler rejects unavailable, malformed, duplicate or
  nonfinite position snapshots and retains the previous local state. Valid empty
  snapshots are still allowed. Failed fetches cannot trigger auto-resolution from
  stale discrepancies.
- Alpaca APIError is handled as a failed fetch. The periodic worker uses an
  interruptible wait, clears its running flag on exit, serializes starts and
  refuses a replacement while the previous thread is alive. Stop wakes idle waits.

Changed runtime files: ai_trading/oms/intent_store.py,
ai_trading/execution/engine.py, ai_trading/execution/position_reconciler.py.
All edits used apply_patch; no broker orders, research trials or gate changes.

## Regression coverage and validation

New tests/unit/test_order_recovery_regressions.py covers four terminal statuses,
late callbacks, two simultaneous database connections, reopen/retry deduplication,
decreasing totals, unavailable/malformed snapshots, API-error recovery and worker
stop/restart. Existing manager doubles and loop tests use the new cumulative-fill
and interruptible-wait contracts. The old artifact in artifacts/audits asserts the
pre-fix bugs intentionally and is historical reproduction evidence, not a test of
the fixed implementation.

- Focused recovery/OMS/manager suite: 38 passed (/tmp/recovery-fix-tests.log).
- Changed-file validator: 247 selected tests passed, lint passed, mypy passed on
  17 source files, compile passed (/tmp/recovery-fix-validation.log).
  Command: PYTHONPYCACHEPREFIX=/tmp/recovery-fix-pycache bash
  scripts/agent_validate_changed.sh --market-hours --skip-runtime-smoke.
- Synthetic OMS lifecycle replay: four scenarios, zero mismatches. Explicit
  isolated SQLite database; /tmp/recovery-fix-parity.log. This is lifecycle
  validation, not an experiment or proof of trading profitability.
- Non-sending tool_runtime_incident_snapshot check passed.

## Risks and rollback

Concurrency was exercised with separate SQLite connections, matching this
deployment. PostgreSQL concurrency was not integration-tested. Transactions can
contend on a busy intent; failures propagate rather than inventing fill evidence.
These fixes do not repair historical duplicate fill rows or certify historical
execution fees. The optional reconciler's current activation remains unverified.
Regular-session operation is still pending; qualification gates remain unchanged.

Rollback only the scoped runtime edits and matching tests, then restart the
service. No database migration or historical data rewrite requires rollback.
Deployment recovery results are recorded in docs/CODEX_HANDOFF.md.
