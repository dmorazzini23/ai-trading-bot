from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.execution.engine import OrderManager
from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.invariants import evaluate_oms_lifecycle_parity_invariants

pytest.importorskip("sqlalchemy")


@pytest.mark.parametrize("initial_status", ["SUBMITTED", "PARTIALLY_FILLED"])
def test_reconcile_missing_open_orders_does_not_fail_open_intent(
    tmp_path,
    initial_status: str,
) -> None:
    store = IntentStore(path=str(tmp_path / "reconcile_missing_open_snapshot.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id=f"intent-missing-open-snapshot-{initial_status.lower()}",
        idempotency_key=f"missing-open-snapshot-{initial_status.lower()}",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-404")
    if initial_status == "PARTIALLY_FILLED":
        store.record_fill(intent.intent_id, fill_qty=1.0, fill_price=190.25)

    summary = manager.reconcile_open_intents(broker_orders=[])

    assert summary["intents_checked"] == 1
    assert summary["marked_failed"] == 0

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == initial_status
    assert refreshed.last_error in (None, "")

    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id in open_intent_ids


def test_reconcile_stale_submitting_intent_fails_closed(
    monkeypatch,
) -> None:
    class _Store:
        def __init__(self, intents):
            self._intents = intents
            self.closed = []

        def get_open_intents(self):
            return list(self._intents)

        def close_intent(self, intent_id: str, *, final_status: str, last_error: str | None = None):
            self.closed.append((intent_id, final_status, last_error))

    intent = SimpleNamespace(
        intent_id="intent-stale-submitting",
        status="SUBMITTING",
        broker_order_id=None,
        updated_at="2026-04-22T00:00:00+00:00",
    )
    store = _Store([intent])
    manager = OrderManager.__new__(OrderManager)
    setattr(manager, "_intent_store", store)

    monkeypatch.setenv("AI_TRADING_OMS_RECONCILE_SUBMIT_STALE_SEC", "1")
    monkeypatch.setattr(
        "ai_trading.execution.engine.safe_utcnow",
        lambda: datetime(2026, 4, 22, 2, 13, 20, tzinfo=UTC),
    )

    summary = manager.reconcile_open_intents(broker_orders=[])

    assert summary["intents_checked"] == 1
    assert summary["marked_failed"] == 1
    assert store.closed == [
        (
            "intent-stale-submitting",
            "FAILED",
            "submit ack missing after 8000s",
        )
    ]


def test_reconcile_submitting_without_timestamp_defers() -> None:
    class _Store:
        def __init__(self, intents):
            self._intents = intents
            self.closed = []

        def get_open_intents(self):
            return list(self._intents)

        def close_intent(self, intent_id: str, *, final_status: str, last_error: str | None = None):
            self.closed.append((intent_id, final_status, last_error))

    intent = SimpleNamespace(
        intent_id="intent-submitting-no-timestamp",
        status="SUBMITTING",
        broker_order_id=None,
        updated_at=None,
    )
    store = _Store([intent])
    manager = OrderManager.__new__(OrderManager)
    setattr(manager, "_intent_store", store)

    summary = manager.reconcile_open_intents(broker_orders=[])

    assert summary["intents_checked"] == 1
    assert summary["deferred_submitting"] == 1
    assert summary["marked_failed"] == 0
    assert store.closed == []


def test_reconcile_stale_pending_submit_intent_fails_closed(
    monkeypatch,
) -> None:
    class _Store:
        def __init__(self, intents):
            self._intents = intents
            self.closed = []

        def get_open_intents(self):
            return list(self._intents)

        def close_intent(self, intent_id: str, *, final_status: str, last_error: str | None = None):
            self.closed.append((intent_id, final_status, last_error))

    intent = SimpleNamespace(
        intent_id="intent-stale-pending-submit",
        status="PENDING_SUBMIT",
        broker_order_id=None,
        updated_at="2026-04-22T00:00:00+00:00",
    )
    store = _Store([intent])
    manager = OrderManager.__new__(OrderManager)
    setattr(manager, "_intent_store", store)

    monkeypatch.setenv("AI_TRADING_OMS_RECONCILE_PENDING_SUBMIT_STALE_SEC", "1")
    monkeypatch.setattr(
        "ai_trading.execution.engine.safe_utcnow",
        lambda: datetime(2026, 4, 22, 2, 13, 20, tzinfo=UTC),
    )

    summary = manager.reconcile_open_intents(broker_orders=[])

    assert summary["intents_checked"] == 1
    assert summary["marked_failed"] == 1
    assert store.closed == [
        (
            "intent-stale-pending-submit",
            "FAILED",
            "submit never claimed after 8000s",
        )
    ]


def test_reconcile_missing_open_orders_closes_intent_on_terminal_broker_lookup(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")
    store = IntentStore(path=str(tmp_path / "reconcile_terminal_lookup.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)

    intent, created = store.create_intent(
        intent_id="intent-terminal-lookup",
        idempotency_key="terminal-lookup-key",
        symbol="MSFT",
        side="buy",
        quantity=5.0,
        status="SUBMITTED",
    )
    assert created is True
    store.mark_submitted(intent.intent_id, "broker-order-505")

    def get_order_by_id(order_id: str) -> dict[str, str]:
        assert order_id == "broker-order-505"
        return {
            "id": order_id,
            "client_order_id": intent.intent_id,
            "status": "filled",
        }

    manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_id_fn=get_order_by_id,
    )

    refreshed = store.get_intent(intent.intent_id)
    assert refreshed is not None
    assert refreshed.status == "FILLED"
    open_intent_ids = {record.intent_id for record in store.get_open_intents()}
    assert intent.intent_id not in open_intent_ids

    event_store = EventStore(path=str(tmp_path / "reconcile_terminal_lookup.db"))
    rows = event_store.list_oms_events(intent_id=intent.intent_id, limit=5000)
    event_store.close()
    event_types = [str(row.get("event_type") or "").strip().upper() for row in rows]
    assert "ORDER_PARTIALLY_FILLED" in event_types
    assert "ORDER_FILLED" in event_types
    assert "INTENT_CLOSED" in event_types

    parity = evaluate_oms_lifecycle_parity_invariants(
        intent_store_path=str(tmp_path / "reconcile_terminal_lookup.db")
    )
    assert int(parity["violations"]["filled_missing_partial_fill"]) == 0


def test_oms_lifecycle_parity_accepts_failed_submit_without_ack(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")
    db_path = tmp_path / "failed_submit_intent.db"
    store = IntentStore(path=str(db_path))
    event_store = EventStore(path=str(db_path))

    intent, created = store.create_intent(
        intent_id="intent-failed-submit",
        idempotency_key="failed-submit-key",
        symbol="AAPL",
        side="sell",
        quantity=1.0,
        status="FAILED",
    )
    assert created is True

    event_store.append_oms_event_payload(
        event_type="SUBMIT_CLAIMED",
        event_source="test",
        idempotency_key="failed-submit-submit-claimed",
        intent_id=intent.intent_id,
        payload={"submit_attempts": 1},
    )
    event_store.append_oms_event_payload(
        event_type="SUBMIT_ATTEMPTED",
        event_source="test",
        idempotency_key="failed-submit-submit-attempted",
        intent_id=intent.intent_id,
        payload={"submit_attempts": 1},
    )
    event_store.append_oms_event_payload(
        event_type="ORDER_FAILED",
        event_source="test",
        idempotency_key="failed-submit-order-failed",
        intent_id=intent.intent_id,
        payload={"final_status": "FAILED", "last_error": "submit ack missing after stale timeout"},
    )
    event_store.append_oms_event_payload(
        event_type="INTENT_CLOSED",
        event_source="test",
        idempotency_key="failed-submit-intent-closed",
        intent_id=intent.intent_id,
        payload={"final_status": "FAILED", "last_error": "submit ack missing after stale timeout"},
    )

    parity = evaluate_oms_lifecycle_parity_invariants(
        intent_store_path=str(db_path)
    )
    assert parity["ok"] is True
    assert int(parity["total_violations"]) == 0
    assert int(parity["violations"]["missing_submit_ack"]) == 0
    assert int(parity["violations"]["terminal_event_mismatch"]) == 0
    assert int(parity["violations"]["terminal_missing_close"]) == 0

    rows = event_store.list_oms_events(intent_id=intent.intent_id, limit=5000)
    event_store.close()
    event_types = [str(row.get("event_type") or "").strip().upper() for row in rows]
    assert "SUBMIT_ACK" not in event_types
    assert "ORDER_FAILED" in event_types
    assert "INTENT_CLOSED" in event_types


def test_oms_lifecycle_parity_accepts_legacy_filled_intent_without_partial_fill(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")
    db_path = tmp_path / "legacy_filled_intent.db"
    store = IntentStore(path=str(db_path))
    event_store = EventStore(path=str(db_path))

    intent, created = store.create_intent(
        intent_id="intent-legacy-filled",
        idempotency_key="legacy-filled-key",
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        status="FILLED",
    )
    assert created is True

    event_store.append_oms_event_payload(
        event_type="SUBMIT_CLAIMED",
        event_source="test",
        idempotency_key="legacy-filled-submit-claimed",
        intent_id=intent.intent_id,
        payload={"submit_attempts": 1},
    )
    event_store.append_oms_event_payload(
        event_type="SUBMIT_ATTEMPTED",
        event_source="test",
        idempotency_key="legacy-filled-submit-attempted",
        intent_id=intent.intent_id,
        payload={"submit_attempts": 1},
    )
    event_store.append_oms_event_payload(
        event_type="SUBMIT_ACK",
        event_source="test",
        idempotency_key="legacy-filled-submit-ack",
        intent_id=intent.intent_id,
        broker_order_id="broker-order-legacy-filled",
        payload={"broker_order_id": "broker-order-legacy-filled", "status": "SUBMITTED"},
    )
    event_store.append_oms_event_payload(
        event_type="ORDER_FILLED",
        event_source="test",
        idempotency_key="legacy-filled-order-fill-1",
        intent_id=intent.intent_id,
        payload={"fill_qty": 1.0, "fill_price": 100.0},
    )
    event_store.append_oms_event_payload(
        event_type="ORDER_FILLED",
        event_source="test",
        idempotency_key="legacy-filled-order-fill-2",
        intent_id=intent.intent_id,
        payload={"fill_qty": 1.0, "fill_price": 100.5},
    )
    event_store.append_oms_event_payload(
        event_type="INTENT_CLOSED",
        event_source="test",
        idempotency_key="legacy-filled-intent-closed",
        intent_id=intent.intent_id,
        payload={"final_status": "FILLED"},
    )

    parity = evaluate_oms_lifecycle_parity_invariants(
        intent_store_path=str(db_path)
    )
    assert parity["ok"] is True
    assert int(parity["total_violations"]) == 0
    assert int(parity["violations"]["filled_missing_partial_fill"]) == 0

    rows = event_store.list_oms_events(intent_id=intent.intent_id, limit=5000)
    event_store.close()
    event_types = [str(row.get("event_type") or "").strip().upper() for row in rows]
    assert event_types.count("ORDER_FILLED") == 2
    assert "ORDER_PARTIALLY_FILLED" not in event_types
