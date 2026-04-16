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
        payload={
            "symbol": "MSFT",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "error": "insufficient_buying_power",
        },
    )
    store.append_oms_event_payload(
        event_type="ORDER_CANCELED",
        event_source="unit_test",
        idempotency_key="cancel-1",
        intent_id="intent-3",
        payload={
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "reason": "operator_cancel",
        },
    )
    store.append_oms_event_payload(
        event_type="ORDER_REJECTED",
        event_source="unit_test",
        idempotency_key="order-reject-1",
        intent_id="intent-4",
        payload={
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "last_error": "hard_risk_block",
        },
    )
    store.append_oms_event_payload(
        event_type="ORDER_FILLED",
        event_source="unit_test",
        idempotency_key="fill-1",
        intent_id="intent-1",
        payload={
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "fill_qty": 2.0,
            "fill_price": 101.0,
            "expected_price": 100.5,
        },
    )
    store.append_oms_event_payload(
        event_type="RECONCILE_UPDATE",
        event_source="unit_test",
        idempotency_key="parent-summary-1",
        intent_id="parent-1",
        payload={
            "record_type": "parent_execution_summary",
            "symbol": "AAPL",
            "strategy_id": "mean_reversion_v2",
            "session_id": "regular_hours",
            "requested_quantity": 10,
            "submitted_quantity": 10,
            "failed_slices": 0,
            "retry_count": 1,
            "cancel_replace_count": 1,
            "success_ratio": 1.0,
            "arrival_slippage_bps_mean": 12.5,
            "arrival_slippage_sample_count": 4,
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
    assert summary["order_cancel_events"] == 1
    assert summary["order_reject_events"] == 1
    assert summary["slippage_sample_count"] == 1
    assert summary["fill_notional"] == pytest.approx(202.0)
    assert summary["cancel_to_submit_ack_rate_pct"] == pytest.approx(100.0)
    assert summary["reject_cancel_rate_pct"] == pytest.approx(150.0)
    assert summary["decision_events_in_window"] == 1
    assert summary["parent_execution_summary_events"] == 1
    assert len(summary["parent_execution_kpis_by_scope"]) == 1
    scope = summary["parent_execution_kpis_by_scope"][0]
    assert scope["symbol"] == "AAPL"
    assert scope["strategy_id"] == "mean_reversion_v2"
    assert scope["session_id"] == "regular_hours"
    assert scope["parent_orders"] == 1
    assert scope["avg_arrival_slippage_bps"] == pytest.approx(12.5)
    reject_reasons = summary["submit_reject_reasons_top"]
    reject_reason_map = {row["reason"]: row["count"] for row in reject_reasons}
    assert reject_reason_map["insufficient_buying_power"] == 1
    assert reject_reason_map["hard_risk_block"] == 1
    cancel_reasons = summary["cancel_reasons_top"]
    assert cancel_reasons[0]["reason"] == "operator_cancel"
    assert cancel_reasons[0]["count"] == 1
    decomposition = summary["realized_slippage_decomposition"]
    assert decomposition["sample_count"] == 1
    assert decomposition["adverse_sample_count"] == 1
    outcomes_by_scope = summary["event_outcomes_by_scope"]
    assert outcomes_by_scope
    top_scope = outcomes_by_scope[0]
    assert top_scope["symbol"] == "AAPL"
    assert top_scope["strategy_id"] == "mean_reversion_v2"
    assert top_scope["session_id"] == "regular_hours"
