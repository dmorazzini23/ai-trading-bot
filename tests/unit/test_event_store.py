from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms.event_store import EventStore
from ai_trading.oms.event_types import DecisionEvent


pytest.importorskip("sqlalchemy")


def test_event_store_enforces_source_idempotency_uniqueness(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'oms_events.db'}")
    inserted_first = store.append_oms_event_payload(
        event_type="INTENT_CREATED",
        event_source="unit_test",
        idempotency_key="same-key",
        intent_id="intent-1",
        payload={"symbol": "AAPL"},
    )
    inserted_second = store.append_oms_event_payload(
        event_type="INTENT_CREATED",
        event_source="unit_test",
        idempotency_key="same-key",
        intent_id="intent-1",
        payload={"symbol": "AAPL"},
    )
    rows = store.list_oms_events(intent_id="intent-1")
    assert inserted_first is True
    assert inserted_second is False
    assert len(rows) == 1
    assert rows[0]["event_type"] == "INTENT_CREATED"


def test_event_store_enforces_decision_idempotency(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'decision_events.db'}")
    decision = DecisionEvent(
        symbol="MSFT",
        decision_action="BUY",
        decision_source="unit_test",
        idempotency_key="decision-key-1",
        strategy_id="strategy-1",
        confidence=0.72,
        expected_edge_bps=12.5,
    )
    inserted_first = store.append_decision_event(decision)
    inserted_second = store.append_decision_event(decision)
    rows = store.list_decision_events(symbol="MSFT")
    assert inserted_first is True
    assert inserted_second is False
    assert len(rows) == 1
    assert rows[0]["symbol"] == "MSFT"
