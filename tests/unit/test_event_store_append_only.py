from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trading.oms.event_store import (
    EventStore,
    _POSTGRES_APPEND_ONLY_GUARD_LOCK_KEY,
)
from ai_trading.oms.event_types import DecisionEvent


pytest.importorskip("sqlalchemy")


def test_oms_events_table_blocks_update_and_delete(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'append_only_oms.db'}")
    inserted = store.append_oms_event_payload(
        event_type="INTENT_CREATED",
        event_source="unit_test",
        idempotency_key="append-only-oms-key",
        intent_id="intent-append-only-1",
        payload={"symbol": "AAPL"},
    )
    assert inserted is True

    with pytest.raises(Exception):  # sqlite raises DBAPI error from trigger abort
        with store._engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE oms_events SET event_type='ORDER_FAILED' WHERE intent_id='intent-append-only-1'"
            )

    with pytest.raises(Exception):  # sqlite raises DBAPI error from trigger abort
        with store._engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM oms_events WHERE intent_id='intent-append-only-1'"
            )

    rows = store.list_oms_events(intent_id="intent-append-only-1")
    assert len(rows) == 1
    assert rows[0]["event_type"] == "INTENT_CREATED"


def test_decision_events_table_blocks_update_and_delete(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'append_only_decision.db'}")
    inserted = store.append_decision_event(
        DecisionEvent(
            symbol="MSFT",
            decision_action="BUY",
            decision_source="unit_test",
            idempotency_key="append-only-decision-key",
            strategy_id="strat-append-only",
            confidence=0.63,
            expected_edge_bps=9.2,
        )
    )
    assert inserted is True

    with pytest.raises(Exception):  # sqlite raises DBAPI error from trigger abort
        with store._engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE decision_events SET decision_action='HOLD' WHERE symbol='MSFT'"
            )

    with pytest.raises(Exception):  # sqlite raises DBAPI error from trigger abort
        with store._engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM decision_events WHERE symbol='MSFT'")

    rows = store.list_decision_events(symbol="MSFT")
    assert len(rows) == 1
    assert rows[0]["decision_action"] == "BUY"


def test_position_snapshots_table_blocks_update_and_delete(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'append_only_positions.db'}")
    inserted = store.append_position_snapshot_payload(
        snapshot_source="unit_test",
        idempotency_key="append-only-position-key",
        snapshot_ts="2026-04-14T00:00:00+00:00",
        symbol="AAPL",
        quantity=2.0,
        market_price=190.0,
        market_value=380.0,
        payload={"symbol": "AAPL"},
    )
    assert inserted is True

    with pytest.raises(Exception):
        with store._engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE position_snapshots SET quantity=3.0 WHERE symbol='AAPL'"
            )

    with pytest.raises(Exception):
        with store._engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM position_snapshots WHERE symbol='AAPL'")

    rows = store.list_position_snapshots(symbol="AAPL")
    assert len(rows) == 1
    assert float(rows[0]["quantity"]) == pytest.approx(2.0)


def test_risk_snapshots_table_blocks_update_and_delete(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'append_only_risk.db'}")
    inserted = store.append_risk_snapshot_payload(
        snapshot_source="unit_test",
        idempotency_key="append-only-risk-key",
        snapshot_ts="2026-04-14T00:00:00+00:00",
        positions_count=2,
        open_orders_count=1,
        payload={"gross_position_qty": 5.0},
    )
    assert inserted is True

    with pytest.raises(Exception):
        with store._engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE risk_snapshots SET positions_count=3 WHERE snapshot_source='unit_test'"
            )

    with pytest.raises(Exception):
        with store._engine.begin() as conn:
            conn.exec_driver_sql(
                "DELETE FROM risk_snapshots WHERE snapshot_source='unit_test'"
            )

    rows = store.list_risk_snapshots(source="unit_test")
    assert len(rows) == 1
    assert int(rows[0]["positions_count"]) == 2


def test_postgres_append_only_guard_bootstrap_uses_advisory_lock() -> None:
    executed: list[str] = []

    class DummyConnection:
        dialect = SimpleNamespace(name="postgresql")

        def execute(self, stmt: object) -> None:
            executed.append(str(stmt).strip())

    store = object.__new__(EventStore)
    store._ensure_append_only_guards(DummyConnection())

    assert executed
    assert executed[0] == (
        f"SELECT pg_advisory_xact_lock({_POSTGRES_APPEND_ONLY_GUARD_LOCK_KEY});"
    )
