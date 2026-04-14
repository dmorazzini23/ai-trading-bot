from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms.event_store import EventStore
from ai_trading.oms.intent_store import IntentStore
from ai_trading.oms.invariants import evaluate_oms_lifecycle_parity_invariants
from ai_trading.oms.simulated_lifecycle import SimulatedLifecycleDriver


pytest.importorskip("sqlalchemy")


def _event_type_stream(store: EventStore, intent_id: str) -> list[str]:
    rows = store.list_oms_events(intent_id=intent_id, limit=5000)
    return [str(row.get("event_type") or "").strip().upper() for row in rows]


def test_simulated_lifecycle_stream_matches_live_intent_store_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "oms_backtest_parity_stream.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    live_store = IntentStore(path=str(db_path))
    live_record, created = live_store.create_intent(
        intent_id="live-intent-1",
        idempotency_key="live-intent-key-1",
        symbol="AAPL",
        side="buy",
        quantity=3.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert live_store.claim_for_submit(live_record.intent_id) is True
    live_store.mark_submitted(live_record.intent_id, "live-broker-1")
    live_store.record_fill(live_record.intent_id, fill_qty=3.0, fill_price=101.25)
    live_store.close_intent(live_record.intent_id, final_status="FILLED")
    live_store.close()

    lifecycle = SimulatedLifecycleDriver(
        enabled=True,
        source="backtest_engine",
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    ref = lifecycle.open_submitted_intent(
        intent_id="bt-intent-1",
        idempotency_key="bt-intent-key-1",
        symbol="AAPL",
        side="buy",
        quantity=3.0,
        decision_ts="2025-01-01T00:00:00+00:00",
        broker_order_id="bt-broker-1",
        strategy_id="backtest_engine",
        metadata={"simulation": True},
    )
    assert ref is not None
    assert lifecycle.record_fill_and_close_intent(
        intent_id=ref.intent_id,
        fill_qty=3.0,
        fill_price=101.25,
        fee=0.0,
        fill_ts="2025-01-01T00:00:01+00:00",
        terminal_status="FILLED",
    )
    lifecycle.close()

    event_store = EventStore(url=f"sqlite:///{db_path}")
    live_stream = _event_type_stream(event_store, "live-intent-1")
    backtest_stream = _event_type_stream(event_store, "bt-intent-1")
    event_store.close()

    assert live_stream
    assert live_stream == backtest_stream


def test_lifecycle_parity_invariants_pass_for_live_and_simulated_intents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "oms_backtest_parity_invariants.db"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_DUAL_WRITE_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "0")

    store = IntentStore(path=str(db_path))
    record, created = store.create_intent(
        intent_id="live-invariant-intent",
        idempotency_key="live-invariant-key",
        symbol="MSFT",
        side="sell",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )
    assert created is True
    assert store.claim_for_submit(record.intent_id) is True
    store.mark_submitted(record.intent_id, "live-invariant-broker")
    store.record_fill(record.intent_id, fill_qty=2.0, fill_price=190.5)
    store.close_intent(record.intent_id, final_status="FILLED")
    store.close()

    lifecycle = SimulatedLifecycleDriver(
        enabled=True,
        source="legacy_backtest_engine",
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    ref = lifecycle.open_submitted_intent(
        intent_id="sim-invariant-intent",
        idempotency_key="sim-invariant-key",
        symbol="MSFT",
        side="sell",
        quantity=2.0,
        decision_ts="2025-01-01T00:00:00+00:00",
    )
    assert ref is not None
    assert lifecycle.record_fill_and_close_intent(
        intent_id=ref.intent_id,
        fill_qty=2.0,
        fill_price=190.5,
        fill_ts="2025-01-01T00:00:01+00:00",
        terminal_status="FILLED",
    )
    lifecycle.close()

    summary = evaluate_oms_lifecycle_parity_invariants(
        database_url=f"sqlite:///{db_path}",
        intent_store_path=str(db_path),
    )
    assert summary["available"] is True
    assert summary["ok"] is True
    assert int(summary["total_violations"]) == 0

