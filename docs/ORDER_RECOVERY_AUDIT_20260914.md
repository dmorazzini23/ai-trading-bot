# Order recovery and reconciliation audit — September 14, 2026

Historical audit: all four findings were subsequently fixed; see
ORDER_RECOVERY_FIXES_20260914.md for validation and deployment status.
This was a bounded source audit of
durable intent transitions, external fill callbacks, and position-reconciliation
failure handling, not a repository-wide certification. No production database,
runtime source, settings, orders or services were modified.

## P1: Terminal intents can become retryable or open again

`ai_trading/oms/intent_store.py:683` updates status after a submission error using
only intent identity as its SQL predicate. A FILLED intent followed by a delayed
error becomes retryable: `claim_for_submit` subsequently succeeds. The manager's
earlier terminal-state read is not atomic with this update. `mark_submitted`
likewise overwrites state without a transition predicate. At line 742,
`record_fill` preserves only FILLED/CLOSED; a late fill on CANCELED becomes
PARTIALLY_FILLED and returns to the open-intent set.

Reproduced both the retry claim and the canceled-intent reopening using isolated
SQLite stores. This establishes incorrect durable state, not an observed duplicate
broker order. Broker client-order idempotency may offer a separate defense.
Fix with database-conditional transitions and preserve terminal cancellation state
while recording late partial fills. Include out-of-order and racing callbacks.

## P1: Concurrent cumulative fill callbacks double-count executions

`ai_trading/execution/engine.py:978–1003` reads the cached/persisted cumulative
quantity, computes a delta and inserts it without one atomic operation. A barrier
between two persisted reads reproduced two callbacks both reporting cumulative
quantity 2 and writing total quantity 4. The store lock serializes inserts but
does not protect the earlier delta calculation. LiveTradingEngine delegates to
this manager through `_sync_durable_order_state` at line 16952.

Fix with a durable atomic cumulative-quantity update or broker execution identity
deduplication. A process-local lock alone does not cover multiple processes or
restarts. Regression coverage must include same-total callbacks, out-of-order
totals, concurrency and recovery. Production occurrence was not established.

## P2: Missing broker snapshots can erase tracked positions

`ai_trading/execution/position_reconciler.py:101` converts a None response into an
empty list and marks the fetch successful. `force_sync_from_broker` then replaces
the local positions with that empty snapshot. Reproduced AAPL quantity 5 becoming
an empty map with `_last_broker_fetch_failed == False`.

Fix by distinguishing validated empty snapshots from unavailable/malformed ones;
retain the last known state and failed-fetch status on invalid responses. This is
a confirmed defect in the exported reconciler surface; this audit did not establish
that the deployed main service currently invokes this optional recovery helper.

## P2: Broker API errors terminate periodic monitoring with a stale running flag

`ai_trading/execution/position_reconciler.py:113` does not catch alpaca-py's APIError,
and the loop at line 215 catches only ValueError/TypeError. Injecting an APIError
escapes the loop while `running` remains True. A subsequent start call would take
the already-running branch despite the worker having exited.

Fix classified broker-error handling, bounded recovery/backoff and truthful worker
lifecycle state. Do not treat an error as an empty successful snapshot. Verify
rate-limit/auth failures, recovery and shutdown/restart. As above, activation of
this optional periodic worker in the deployed main service was not established.

## Reproduction and existing coverage

`./venv/bin/pytest -q /tmp/test_order_recovery_audit.py`: 5 passed. These assertions
deliberately prove the defective behavior; they are not fix-regression tests.
The identical fixture is retained at
`artifacts/audits/test_order_recovery_20260914.py`. Uses local SQLite and fake broker
responses; no network or real orders. Log: `/tmp/order-recovery-audit-tests.log`.

Existing focused tests: 17 passed across `test_intent_store_terminal_statuses.py`,
`test_intent_store_idempotency.py`, `test_position_reconciler_phase2_local_78b.py`
and `test_phase2_agent_execution_runtime.py`. Log:
`/tmp/order-recovery-existing-tests.log`. Their success does not cover the above
failure cases. No deployment or live smoke check was necessary for this read-only
audit. Preserve existing research and trading gates when implementing fixes.
