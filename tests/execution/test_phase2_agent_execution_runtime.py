from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.execution import engine as eng
from ai_trading.execution import live_trading as lt
from ai_trading.execution.position_reconciler import (
    PositionDiscrepancy,
    PositionReconciler,
)


class _BrokerBackedReconciler(PositionReconciler):
    def __init__(self, snapshots: list[dict[str, float]]) -> None:
        super().__init__(api_client=object())
        self._snapshots = list(snapshots)

    def get_broker_positions(self) -> dict[str, float]:
        broker_positions = self._snapshots.pop(0) if self._snapshots else {}
        with self._lock:
            self._broker_positions = broker_positions.copy()
        return broker_positions


def test_position_reconciler_classifies_severity_and_auto_resolves_safe_drift() -> None:
    reconciler = _BrokerBackedReconciler(
        [
            {
                "MSFT": 3.0,
                "GOOG": -2.0,
                "TSLA": 8.0,
                "NFLX": 12.0,
            }
        ]
    )
    reconciler.large_discrepancy_threshold = 10
    reconciler.update_bot_position("AAPL", 5.0, reason="seed")
    reconciler.update_bot_position("GOOG", 2.0, reason="seed")
    reconciler.update_bot_position("TSLA", 2.0, reason="seed")
    reconciler.update_bot_position("NFLX", 0.0, reason="seed")

    discrepancies = {
        discrepancy.symbol: discrepancy for discrepancy in reconciler.reconcile_positions()
    }

    assert discrepancies["AAPL"].discrepancy_type == "phantom_position"
    assert discrepancies["MSFT"].discrepancy_type == "missing_position"
    assert discrepancies["GOOG"].discrepancy_type == "direction_mismatch"
    assert discrepancies["TSLA"].discrepancy_type == "quantity_mismatch"
    assert discrepancies["NFLX"].severity == "high"
    assert discrepancies["TSLA"].severity == "medium"
    assert discrepancies["MSFT"].to_dict()["difference"] == pytest.approx(3.0)

    resolved = reconciler.auto_resolve_discrepancies(list(discrepancies.values()))

    assert resolved == 4
    assert reconciler.get_bot_positions() == {
        "AAPL": 0,
        "GOOG": -2.0,
        "TSLA": 8.0,
        "NFLX": 0.0,
        "MSFT": 3.0,
    }
    assert len(reconciler.get_current_discrepancies()) == 5
    stats = reconciler.get_reconciliation_stats()
    assert stats["current_discrepancies"] == 5
    assert stats["severity_breakdown"] == {"medium": 4, "high": 1}
    assert stats["avg_discrepancies_per_reconciliation"] == 5.0


def test_position_reconciler_bounds_histories_and_force_syncs_from_broker() -> None:
    reconciler = _BrokerBackedReconciler([{"AAPL": 3.0}, {"MSFT": 4.0}])
    reconciler.update_bot_position("AAPL", 1.0)
    reconciler._discrepancy_history = [
        PositionDiscrepancy(f"S{i}", 0.0, 1.0, "missing_position", "low")
        for i in range(1000)
    ]
    reconciler._reconciliation_history = [
        {"timestamp": str(i), "discrepancies_count": i % 3} for i in range(100)
    ]

    discrepancies = reconciler.reconcile_positions()

    assert [discrepancy.symbol for discrepancy in discrepancies] == ["AAPL"]
    assert len(reconciler.get_discrepancy_history(limit=1000)) == 500
    assert reconciler.get_discrepancy_history("AAPL", limit=1000)[0].symbol == "AAPL"
    assert len(reconciler.get_reconciliation_history(limit=1000)) == 50

    synced = reconciler.force_sync_from_broker()

    assert synced == {"MSFT": 4.0}
    assert reconciler.get_bot_positions() == {"MSFT": 4.0}


class _FakeIntentStore:
    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}
        self.claimed: list[tuple[str, int]] = []
        self.submit_errors: list[tuple[str, str]] = []
        self.fills: dict[str, list[SimpleNamespace]] = {}
        self.closed: list[tuple[str, str, str | None]] = []
        self.raise_on_claim = False

    def create_intent(self, **kwargs: Any) -> tuple[SimpleNamespace, bool]:
        record = SimpleNamespace(
            intent_id=kwargs["intent_id"],
            idempotency_key=kwargs["idempotency_key"],
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            quantity=kwargs["quantity"],
            status=kwargs.get("status", "PENDING_SUBMIT"),
            broker_order_id=None,
            last_error=None,
        )
        self.records[record.intent_id] = record
        return record, True

    def claim_for_submit(self, intent_id: str, *, stale_after_seconds: int) -> None:
        self.claimed.append((intent_id, stale_after_seconds))
        if self.raise_on_claim:
            raise RuntimeError("claim unavailable")
        self.records[intent_id].status = "SUBMITTING"

    def get_intent(self, intent_id: str) -> SimpleNamespace | None:
        return self.records.get(intent_id)

    def mark_submitted(self, intent_id: str, broker_order_id: str) -> None:
        self.records[intent_id].broker_order_id = broker_order_id
        self.records[intent_id].status = "SUBMITTED"

    def list_fills(self, intent_id: str) -> list[SimpleNamespace]:
        return list(self.fills.get(intent_id, []))

    def record_fill(
        self,
        intent_id: str,
        *,
        fill_qty: float,
        fill_price: float | None,
    ) -> None:
        self.fills.setdefault(intent_id, []).append(
            SimpleNamespace(fill_qty=fill_qty, fill_price=fill_price)
        )
        self.records[intent_id].status = "PARTIALLY_FILLED"

    def record_submit_error(self, intent_id: str, error: str) -> None:
        self.submit_errors.append((intent_id, error))
        self.records[intent_id].last_error = error

    def close_intent(
        self,
        intent_id: str,
        *,
        final_status: str,
        last_error: str | None = None,
    ) -> None:
        self.closed.append((intent_id, final_status, last_error))
        self.records[intent_id].status = final_status
        self.records[intent_id].last_error = last_error


def _order_manager_with_store(store: _FakeIntentStore) -> eng.OrderManager:
    manager = eng.OrderManager.__new__(eng.OrderManager)
    manager._intent_store = store
    manager._intent_by_order_id = {}
    manager._intent_reported_fill_qty = {}
    return manager


def test_order_manager_external_lifecycle_records_delta_fills_and_clears_terminal_maps() -> None:
    store = _FakeIntentStore()
    manager = _order_manager_with_store(store)

    assert manager.begin_external_order_lifecycle(
        intent_id="intent-1",
        idempotency_key="idem-1",
        symbol="aapl",
        side="BUY",
        quantity=5,
        stale_after_seconds=0,
        metadata={"source": "test"},
    ) == "intent-1"
    assert store.claimed == [("intent-1", 1)]
    assert manager.record_external_submit_error(
        order_id="intent-1",
        error="temporary broker outage",
    ) == "intent-1"
    assert store.submit_errors == [("intent-1", "temporary broker outage")]

    assert manager.sync_external_order_state(
        intent_id="intent-1",
        order_id="broker-1",
        client_order_id="client-1",
        status="accepted",
        filled_qty="2.5",
        fill_price="101.25",
    ) == "intent-1"
    assert [(fill.fill_qty, fill.fill_price) for fill in store.list_fills("intent-1")] == [
        (2.5, 101.25)
    ]

    manager.sync_external_order_state(
        client_order_id="client-1",
        status="filled",
        filled_qty=5,
        fill_price="102.5",
    )

    assert [(fill.fill_qty, fill.fill_price) for fill in store.list_fills("intent-1")] == [
        (2.5, 101.25),
        (2.5, 102.5),
    ]
    assert store.closed == [("intent-1", "FILLED", None)]
    assert manager._intent_by_order_id == {}
    assert manager._intent_reported_fill_qty == {}


def test_order_manager_external_lifecycle_returns_none_when_claim_fails() -> None:
    store = _FakeIntentStore()
    store.raise_on_claim = True
    manager = _order_manager_with_store(store)

    assert (
        manager.begin_external_order_lifecycle(
            intent_id="intent-claim-fail",
            idempotency_key="idem-claim-fail",
            symbol="aapl",
            side="BUY",
            quantity=5,
            metadata={"source": "test"},
        )
        is None
    )
    assert store.claimed == [("intent-claim-fail", 90)]
    assert manager._intent_by_order_id == {}
    assert manager._intent_reported_fill_qty == {}


def test_engine_parent_summary_event_handles_missing_store_missing_id_and_store_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = eng.ExecutionEngine()

    monkeypatch.setattr(engine, "_resolve_execution_audit_store", lambda: None)
    assert engine._emit_parent_execution_summary_event({"parent_order_id": "parent-1"}) is False

    class _Store:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.calls: list[dict[str, Any]] = []

        def append_oms_event_payload(self, **kwargs: Any) -> bool:
            if self.fail:
                raise RuntimeError("event store down")
            self.calls.append(dict(kwargs))
            return True

    store = _Store()
    monkeypatch.setattr(engine, "_resolve_execution_audit_store", lambda: store)
    assert engine._emit_parent_execution_summary_event({"symbol": "AAPL"}) is False

    assert engine._emit_parent_execution_summary_event(
        {
            "parent_order_id": "parent-1",
            "symbol": "aapl",
            "strategy_id": "mean",
            "session_id": "regular",
            "submitted_slices": 2,
            "failed_slices": 0,
        }
    ) is True
    assert store.calls[0]["intent_id"] == "parent-1"
    assert store.calls[0]["payload"]["record_type"] == "parent_execution_summary"
    assert store.calls[0]["payload"]["kpi_scope"] == {
        "symbol": "AAPL",
        "strategy_id": "mean",
        "session_id": "regular",
    }

    broken_store = _Store(fail=True)
    monkeypatch.setattr(engine, "_resolve_execution_audit_store", lambda: broken_store)
    assert engine._emit_parent_execution_summary_event({"parent_order_id": "parent-2"}) is False


def test_live_trading_small_helpers_handle_mapping_payloads_and_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lt, "_config_int", lambda _name, _default: -5)
    assert lt._fallback_limit_buffer_bps() == 0
    monkeypatch.setattr(lt, "_config_int", lambda _name, _default: 12)
    assert lt._fallback_limit_buffer_bps() == 12

    assert lt._extract_cash_balance(None) is None
    assert lt._extract_cash_balance(
        SimpleNamespace(cash="", cash_balance="bad", buying_power="250.50")
    ) == pytest.approx(250.5)
    assert lt._extract_cash_balance(
        {"cash": "", "cash_balance": "oops", "available_cash": "19.75"}
    ) == pytest.approx(19.75)

    positions = [
        SimpleNamespace(symbol="aapl", qty="2.5"),
        {"asset_symbol": "msft", "quantity": "3"},
        {"symbol": "bad", "qty": "not-a-number"},
        SimpleNamespace(symbol="", qty=10),
    ]

    assert lt._positions_to_quantity_map(positions) == {"AAPL": 2.5, "MSFT": 3.0}
