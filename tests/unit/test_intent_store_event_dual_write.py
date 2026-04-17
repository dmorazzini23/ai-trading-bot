from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore


pytest.importorskip("sqlalchemy")


def test_intent_store_event_dual_write_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "intent_event_default.db"
    monkeypatch.delenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    store = IntentStore(path=str(db_path))
    record, created = store.create_intent(
        intent_id="intent-default-1",
        idempotency_key="intent-default-key-1",
        symbol="AAPL",
        side="buy",
        quantity=5.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert store.claim_for_submit(record.intent_id) is True
    store.mark_submitted(record.intent_id, "broker-default-1")
    store.record_fill(record.intent_id, fill_qty=5.0, fill_price=201.5)
    store.close_intent(record.intent_id, final_status="FILLED")
    store.close()

    event_store = EventStore(url=f"sqlite:///{db_path}")
    events = event_store.list_oms_events(intent_id=record.intent_id)
    event_store.close()
    assert events == []


def test_intent_store_dual_writes_lifecycle_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "intent_event_dual_write.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    store = IntentStore(path=str(db_path))
    record, created = store.create_intent(
        intent_id="intent-dual-1",
        idempotency_key="intent-dual-key-1",
        symbol="AAPL",
        side="buy",
        quantity=5.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert store.claim_for_submit(record.intent_id) is True
    store.mark_submitted(record.intent_id, "broker-dual-1")
    store.record_fill(record.intent_id, fill_qty=5.0, fill_price=201.5)
    store.close_intent(record.intent_id, final_status="FILLED")
    store.close()

    event_store = EventStore(url=f"sqlite:///{db_path}")
    events = event_store.list_oms_events(intent_id=record.intent_id)
    event_store.close()
    event_types = {str(item["event_type"]) for item in events}
    assert "INTENT_CREATED" in event_types
    assert "SUBMIT_CLAIMED" in event_types
    assert "SUBMIT_ATTEMPTED" in event_types
    assert "SUBMIT_ACK" in event_types
    assert "ORDER_PARTIALLY_FILLED" in event_types
    assert "ORDER_FILLED" in event_types
    assert "INTENT_CLOSED" in event_types
