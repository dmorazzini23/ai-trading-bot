from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms.event_store import EventStore
from ai_trading.oms.event_types import DecisionEvent
from ai_trading.tca.event_analytics import summarize_oms_event_tca


pytest.importorskip("sqlalchemy")


def test_summarize_oms_event_tca_from_immutable_events(tmp_path: Path) -> None:
    db_path = tmp_path / "oms_tca_event_analytics.db"
    store = EventStore(url=f"sqlite:///{db_path}")
    store.append_oms_event_payload(
        event_type="SUBMIT_ACK",
        event_source="unit_test",
        idempotency_key="ack-1",
        intent_id="intent-1",
        payload={"symbol": "AAPL"},
    )
    store.append_oms_event_payload(
        event_type="SUBMIT_REJECT",
        event_source="unit_test",
        idempotency_key="reject-1",
        intent_id="intent-2",
        payload={"symbol": "MSFT", "error": "insufficient_buying_power"},
    )
    store.append_oms_event_payload(
        event_type="ORDER_FILLED",
        event_source="unit_test",
        idempotency_key="fill-1",
        intent_id="intent-1",
        payload={
            "symbol": "AAPL",
            "fill_qty": 2.0,
            "fill_price": 101.0,
            "expected_price": 100.5,
        },
    )
    store.append_decision_event(
        event=DecisionEvent(
            symbol="AAPL",
            decision_action="BUY",
            decision_source="unit_test",
            idempotency_key="decision-1",
            confidence=0.7,
            expected_edge_bps=12.0,
        )
    )
    store.close()

    summary = summarize_oms_event_tca(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
        limit=1000,
    )
    assert summary["filled_events"] == 1
    assert summary["submit_ack_events"] == 1
    assert summary["submit_reject_events"] == 1
    assert summary["slippage_sample_count"] == 1
    assert summary["fill_notional"] == pytest.approx(202.0)
    assert summary["decision_events_in_window"] == 1
