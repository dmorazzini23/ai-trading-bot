from __future__ import annotations

from pathlib import Path

import pytest

from ai_trading.oms import event_store as event_store_module
from ai_trading.oms.event_store import EventStore
from ai_trading.oms.event_types import DecisionEvent


pytest.importorskip("sqlalchemy")


def test_event_store_jsonl_fallback_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "oms_events.jsonl"
    monkeypatch.delenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", raising=False)
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_PATH", str(jsonl_path))

    store = EventStore(url=f"sqlite:///{tmp_path / 'oms_events_default.db'}")
    inserted = store.append_oms_event_payload(
        event_type="INTENT_CREATED",
        event_source="unit_test",
        idempotency_key="jsonl-default-off-key",
        intent_id="intent-default-off-1",
        payload={"symbol": "AAPL"},
    )

    assert inserted is True
    assert jsonl_path.exists() is False


def test_event_store_can_opt_into_jsonl_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "oms_events_opt_in.jsonl"
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_OMS_EVENT_JSONL_PATH", str(jsonl_path))

    store = EventStore(url=f"sqlite:///{tmp_path / 'oms_events_opt_in.db'}")
    inserted = store.append_oms_event_payload(
        event_type="INTENT_CREATED",
        event_source="unit_test",
        idempotency_key="jsonl-opt-in-key",
        intent_id="intent-opt-in-1",
        payload={"symbol": "AAPL"},
    )

    assert inserted is True
    assert jsonl_path.exists() is True


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


def test_event_store_position_snapshots_are_idempotent(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'position_snapshots.db'}")
    inserted_first = store.append_position_snapshot_payload(
        snapshot_source="unit_test",
        idempotency_key="position-key-1",
        snapshot_ts="2026-04-14T00:00:00+00:00",
        symbol="AAPL",
        quantity=5.0,
        side="long",
        market_price=190.5,
        market_value=952.5,
        payload={"symbol": "AAPL", "quantity": 5.0},
    )
    inserted_second = store.append_position_snapshot_payload(
        snapshot_source="unit_test",
        idempotency_key="position-key-1",
        snapshot_ts="2026-04-14T00:00:00+00:00",
        symbol="AAPL",
        quantity=5.0,
        side="long",
        market_price=190.5,
        market_value=952.5,
        payload={"symbol": "AAPL", "quantity": 5.0},
    )
    rows = store.list_position_snapshots(symbol="AAPL")
    assert inserted_first is True
    assert inserted_second is False
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert float(rows[0]["quantity"]) == pytest.approx(5.0)


def test_event_store_risk_snapshots_are_idempotent(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'risk_snapshots.db'}")
    inserted_first = store.append_risk_snapshot_payload(
        snapshot_source="unit_test",
        idempotency_key="risk-key-1",
        snapshot_ts="2026-04-14T00:00:00+00:00",
        exposure_pct=0.24,
        positions_count=3,
        open_orders_count=2,
        payload={"gross_position_qty": 7.0, "net_position_qty": 3.0},
    )
    inserted_second = store.append_risk_snapshot_payload(
        snapshot_source="unit_test",
        idempotency_key="risk-key-1",
        snapshot_ts="2026-04-14T00:00:00+00:00",
        exposure_pct=0.24,
        positions_count=3,
        open_orders_count=2,
        payload={"gross_position_qty": 7.0, "net_position_qty": 3.0},
    )
    rows = store.list_risk_snapshots(source="unit_test")
    assert inserted_first is True
    assert inserted_second is False
    assert len(rows) == 1
    assert int(rows[0]["positions_count"]) == 3
    assert int(rows[0]["open_orders_count"]) == 2


def test_event_store_migration_status_handles_missing_alembic_table(tmp_path: Path) -> None:
    store = EventStore(url=f"sqlite:///{tmp_path / 'migration_status.db'}")

    payload = store.migration_status(expected_revision="20260414_0001")
    health = store.is_healthy(expected_revision="20260414_0001")

    assert payload["expected_revision"] == "20260414_0001"
    assert payload["available"] is False
    assert payload["managed"] is False
    assert payload["current_revision"] is None
    assert payload["at_head"] is None
    assert payload["reason"] == "not_alembic_managed"
    assert health["connected"] is True
    assert health["ok"] is False
    assert health["migration"]["reason"] == "not_alembic_managed"


def test_event_store_health_reads_rollback_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    store = EventStore(url="sqlite://")

    class _DummyResult:
        def __init__(self, row):
            self._row = row
            self.closed = False

        def first(self):
            return self._row

        def close(self):
            self.closed = True

    class _DummyConnection:
        def __init__(self):
            self.rollback_count = 0
            self._rows = [("rev-1",), None, ("rev-1",)]
            self.results: list[_DummyResult] = []

        def execute(self, _stmt):
            row = self._rows.pop(0) if self._rows else None
            result = _DummyResult(row)
            self.results.append(result)
            return result

        def rollback(self):
            self.rollback_count += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_conn = _DummyConnection()

    class _DummyInspector:
        @staticmethod
        def has_table(_name: str) -> bool:
            return True

    class _DummyEngine:
        def connect(self):
            return dummy_conn

    monkeypatch.setattr(store, "_engine", _DummyEngine())
    monkeypatch.setattr(event_store_module, "inspect", lambda _engine: _DummyInspector())

    migration = store.migration_status(expected_revision="rev-1")
    health = store.is_healthy(expected_revision="rev-1")

    assert migration["available"] is True
    assert migration["managed"] is True
    assert migration["current_revision"] == "rev-1"
    assert health["connected"] is True
    assert health["ok"] is True
    assert dummy_conn.rollback_count == 3
    assert all(result.closed for result in dummy_conn.results)


def test_event_store_health_fails_closed_on_sqlalchemy_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EventStore(url="sqlite://")

    class _BrokenEngine:
        def connect(self):
            raise event_store_module.SQLAlchemyError("database offline")

    monkeypatch.setattr(store, "_engine", _BrokenEngine())

    health = store.is_healthy(expected_revision="rev-1")

    assert health["ok"] is False
    assert health["connected"] is False
    assert health["error"] == "database offline"
