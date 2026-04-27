from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ai_trading.oms.event_types import DecisionEvent, OmsEvent
from ai_trading.oms import invariants


def test_oms_event_contracts_reject_missing_identity_fields() -> None:
    with pytest.raises(ValueError, match="idempotency_key"):
        OmsEvent(
            event_type="INTENT_CREATED",
            event_source="",
            idempotency_key=" ",
            payload={},
        ).normalized()

    with pytest.raises(ValueError, match="symbol"):
        DecisionEvent(
            symbol=" ",
            decision_action="BUY",
            decision_source="strategy",
            idempotency_key="key",
        ).normalized()

    with pytest.raises(ValueError, match="idempotency_key"):
        DecisionEvent(
            symbol="AAPL",
            decision_action="BUY",
            decision_source="strategy",
            idempotency_key=" ",
        ).normalized()


def test_oms_invariants_record_reconciliation_and_lifecycle_violations(monkeypatch) -> None:
    intents = [
        SimpleNamespace(intent_id="submitted", status="SUBMITTED", broker_order_id="broker-1"),
        SimpleNamespace(intent_id="filled", status="FILLED", broker_order_id="broker-2"),
    ]
    event_rows = {
        "submitted": [{"event_type": "SUBMIT_ACK"}],
        "filled": [{"event_type": "INTENT_CREATED"}, {"event_type": "ORDER_CANCELED"}],
    }
    closed: list[str] = []

    class FakeIntentStore:
        def __init__(self, *_, **__) -> None:
            pass

        def list_intents(self, *, limit: int):
            assert limit == 1
            return intents

        def close(self) -> None:
            closed.append("intent")
            raise RuntimeError("close failed")

    class FakeEventStore:
        def __init__(self, *_, **__) -> None:
            pass

        def list_oms_events(self, *, intent_id: str, limit: int):
            assert limit == 5000
            return event_rows[intent_id]

        def close(self) -> None:
            closed.append("event")
            raise RuntimeError("close failed")

    monkeypatch.setitem(
        sys.modules,
        "ai_trading.oms.intent_store",
        SimpleNamespace(IntentStore=FakeIntentStore),
    )
    monkeypatch.setitem(
        sys.modules,
        "ai_trading.oms.event_store",
        SimpleNamespace(EventStore=FakeEventStore),
    )

    reconciliation = invariants.evaluate_oms_reconciliation_invariants(limit=0)
    parity = invariants.evaluate_oms_lifecycle_parity_invariants(limit=0)

    assert reconciliation["available"] is True
    assert reconciliation["ok"] is False
    assert reconciliation["scanned_intents"] == 2
    assert reconciliation["violations"]["missing_intent_created"] == 1
    assert reconciliation["violations"]["submitted_missing_ack"] == 1
    assert reconciliation["violations"]["terminal_missing_close"] == 1
    assert reconciliation["violations"]["filled_missing_fill_event"] == 1
    assert {sample["issue"] for sample in reconciliation["sample_violations"]} >= {
        "missing_intent_created",
        "filled_missing_fill_event",
    }

    assert parity["available"] is True
    assert parity["ok"] is False
    assert parity["violations"]["missing_submit_claim"] == 2
    assert parity["violations"]["missing_submit_attempt"] == 2
    assert parity["violations"]["terminal_event_mismatch"] == 1
    assert parity["violations"]["terminal_missing_close"] == 1
    assert parity["violations"]["filled_missing_partial_fill"] == 1
    assert closed == ["intent", "event", "intent", "event"]
