# Sprint 1 ADR: Durable OMS Persistence Foundation

Status: Accepted  
Date: 2026-04-14

## Context

The repository currently mixes:

- Durable SQL persistence for OMS intents/fills in `ai_trading/oms/intent_store.py`
- JSONL append-only idempotency ledger in `ai_trading/oms/ledger.py`
- File-based trade history persistence in `ai_trading/meta_learning/persistence.py`
- Legacy database stubs in `ai_trading/database/connection.py`

This produces uneven state durability and weak auditability for a live system.

## Decision

Sprint 1 establishes a migration-managed Postgres-first durability layer while preserving safe fallback behavior:

1. Introduce Alembic-managed schema evolution.
2. Keep existing `intents` and `intent_fills` as canonical OMS lifecycle tables.
3. Add immutable append-only event tables:
   - `oms_events`
   - `decision_events`
4. Use `DATABASE_URL` as authoritative database URL for live mode.
5. Keep JSONL ledger as emergency fallback only; it is not the live system of record.
6. Ensure health endpoints can report DB readiness without uncaught exceptions.

## Event Taxonomy (Sprint 1)

`oms_events.event_type` initial controlled vocabulary:

- `DECISION_EMITTED`
- `INTENT_CREATED`
- `SUBMIT_CLAIMED`
- `SUBMIT_ATTEMPTED`
- `SUBMIT_ACK`
- `SUBMIT_REJECT`
- `ORDER_PARTIALLY_FILLED`
- `ORDER_FILLED`
- `ORDER_CANCELED`
- `ORDER_FAILED`
- `INTENT_CLOSED`
- `RECONCILE_UPDATE`

`decision_events.decision_action` initial controlled vocabulary:

- `BUY`
- `SELL`
- `HOLD`
- `REDUCE`
- `EXIT`

## Idempotency and Ordering Rules

1. Every persisted event must include an idempotency key.
2. Idempotency uniqueness is enforced by `(event_source, idempotency_key)` in `oms_events`.
3. Intent lifecycle order is represented by `sequence_no` plus `event_ts`.
4. Writers must be retry-safe; duplicate writes must not create duplicate events.

## Non-Goals (Sprint 1)

- Full multi-venue execution router redesign.
- Full live/backtest path unification.
- Full control-plane UX.

## Rollout

1. Apply migrations.
2. Dual-write lifecycle events to DB + current fallback ledger.
3. Validate with integration tests and restart reconciliation tests.
4. Gate live rollout on health/readiness checks.

## Risks and Mitigations

- Risk: Existing deployments already contain `intents` tables created outside migrations.
  - Mitigation: migration creates tables conditionally when missing.
- Risk: DB outage at runtime.
  - Mitigation: explicit fallback path with structured warning logs.
- Risk: Schema drift.
  - Mitigation: enforce migrations in CI and startup readiness checks.

## Follow-up

- Sprint 2: centralize order lifecycle state transitions through immutable events.
- Sprint 3: route backtests through shared lifecycle state machine.
