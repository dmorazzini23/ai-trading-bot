"""Isolated process-crash tests for durable OMS and broker reconciliation."""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest

from ai_trading.execution.engine import OrderManager
from ai_trading.execution.live_trading import ExecutionEngine
from ai_trading.oms.intent_store import IntentStore

pytest.importorskip("sqlalchemy")


class _BrokerSimulator:
    """Tiny durable broker truth for fault tests; never contacts a broker."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS orders ("
                "client_order_id TEXT PRIMARY KEY, id TEXT NOT NULL, "
                "status TEXT NOT NULL, filled_qty REAL NOT NULL)"
            )

    def accept(self, client_order_id: str) -> dict[str, object]:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO orders VALUES (?, ?, 'accepted', 0)",
                (client_order_id, f"broker-{client_order_id}"),
            )
        order = self.lookup(client_order_id)
        assert order is not None
        return order

    def set_fill(self, client_order_id: str, quantity: float, *, terminal: bool) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE orders SET status = ?, filled_qty = ? WHERE client_order_id = ?",
                ("filled" if terminal else "partially_filled", quantity, client_order_id),
            )

    def cancel(self, client_order_id: str) -> bool:
        with sqlite3.connect(self.path) as connection:
            result = connection.execute(
                "UPDATE orders SET status = 'canceled' "
                "WHERE client_order_id = ? AND status != 'filled'",
                (client_order_id,),
            )
        return result.rowcount > 0

    def lookup(self, client_order_id: str) -> dict[str, object] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT id, client_order_id, status, filled_qty FROM orders "
                "WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(zip(("id", "client_order_id", "status", "filled_qty"), row))

    def count(self) -> int:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM orders").fetchone()
        assert row is not None
        return int(row[0])


def _accept_then_crash(intent_db: str, broker_db: str, intent_id: str) -> None:
    store = IntentStore(path=intent_db)
    if not store.claim_for_submit(intent_id):
        os._exit(41)
    _BrokerSimulator(Path(broker_db)).accept(intent_id)
    os._exit(47)  # Broker accepted; local acknowledgement was never persisted.


def _accept_then_ack_write_fails(intent_db: str, broker_db: str, intent_id: str) -> None:
    from sqlalchemy import event

    store = IntentStore(path=intent_db)
    if not store.claim_for_submit(intent_id):
        os._exit(41)
    broker_order = _BrokerSimulator(Path(broker_db)).accept(intent_id)

    @event.listens_for(store._engine, "before_cursor_execute")
    def fail_ack_write(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("UPDATE INTENTS"):
            raise OSError("simulated acknowledgement storage failure")

    try:
        store.mark_submitted(intent_id, str(broker_order["id"]))
    except OSError:
        os._exit(53)
    os._exit(54)


def _start_crashed_submission(
    tmp_path: Path, *, storage_failure: bool = False
) -> tuple[IntentStore, _BrokerSimulator, str]:
    intent_db = tmp_path / "intents.db"
    broker_db = tmp_path / "broker.db"
    store = IntentStore(path=str(intent_db))
    broker = _BrokerSimulator(broker_db)
    intent_id = "client-process-fault-1"
    intent, created = store.create_intent(
        intent_id=intent_id,
        idempotency_key="process-fault-key",
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        status="PENDING_SUBMIT",
    )
    assert created
    assert intent.intent_id == intent_id
    process = multiprocessing.get_context("fork").Process(
        target=_accept_then_ack_write_fails if storage_failure else _accept_then_crash,
        args=(str(intent_db), str(broker_db), intent_id),
    )
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("fault worker did not terminate")
    assert process.exitcode == (53 if storage_failure else 47)
    assert broker.count() == 1
    return store, broker, intent_id


def test_restart_after_broker_acceptance_does_not_resubmit(tmp_path: Path) -> None:
    store, broker, intent_id = _start_crashed_submission(tmp_path)
    manager = OrderManager()
    manager.configure_intent_store(store)

    assert not store.claim_for_submit(intent_id, stale_after_seconds=1)
    summary = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=broker.lookup,
    )

    recovered = store.get_intent(intent_id)
    assert recovered is not None
    assert recovered.status == "SUBMITTED"
    assert recovered.broker_order_id == f"broker-{intent_id}"
    assert recovered.submit_attempts == 1
    assert summary["marked_submitted"] == 1
    assert broker.count() == 1


def test_existing_durable_submit_claim_cannot_begin_second_broker_attempt(
    tmp_path: Path,
) -> None:
    store, broker, intent_id = _start_crashed_submission(tmp_path)
    manager = OrderManager()
    manager.configure_intent_store(store)

    assert manager.begin_external_order_lifecycle(
        intent_id=intent_id,
        idempotency_key="process-fault-key",
        symbol="AAPL",
        side="buy",
        quantity=2.0,
    ) is None
    existing = store.get_intent(intent_id)
    assert existing is not None
    assert existing.status == "SUBMITTING"
    assert existing.submit_attempts == 1
    assert broker.count() == 1


def test_unknown_broker_outcome_blocks_new_opening_until_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, broker, intent_id = _start_crashed_submission(tmp_path)
    manager = OrderManager()
    manager.configure_intent_store(store)
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine.order_manager = manager
    engine.execution_mode = "paper"
    engine._broker_sync = None
    monkeypatch.setenv("AI_TRADING_EXECUTION_PHASE_GATE_ENABLED", "0")

    unresolved = manager.reconcile_open_intents(broker_orders=[])
    assert unresolved["deferred_submitting"] == 1
    assert store.get_intent(intent_id).status == "SUBMITTING"
    assert engine._execution_phase_allows_submits(closing_position=False) == (
        False,
        "oms_submit_outcome_unresolved",
    )
    assert engine._execution_phase_allows_submits(closing_position=True) == (True, None)

    manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=broker.lookup,
    )
    assert store.get_intent(intent_id).status == "SUBMITTED"
    assert engine._execution_phase_allows_submits(closing_position=False) == (True, None)
    assert broker.count() == 1


def test_ack_storage_failure_survives_restart_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, broker, intent_id = _start_crashed_submission(tmp_path, storage_failure=True)
    store = IntentStore(path=str(tmp_path / "intents.db"))
    manager = OrderManager()
    manager.configure_intent_store(store)
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine.order_manager = manager
    engine.execution_mode = "paper"
    engine._broker_sync = None
    monkeypatch.setenv("AI_TRADING_EXECUTION_PHASE_GATE_ENABLED", "0")

    unresolved = store.get_intent(intent_id)
    assert unresolved is not None and unresolved.status == "SUBMITTING"
    assert not store.claim_for_submit(intent_id, stale_after_seconds=1)
    assert engine._execution_phase_allows_submits(closing_position=False) == (
        False,
        "oms_submit_outcome_unresolved",
    )
    manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=broker.lookup,
    )
    recovered = store.get_intent(intent_id)
    assert recovered is not None and recovered.status == "SUBMITTED"
    assert engine._execution_phase_allows_submits(closing_position=False) == (True, None)
    assert broker.count() == 1


def test_fill_and_cancel_race_after_disconnect_preserves_fill(tmp_path: Path) -> None:
    store, broker, intent_id = _start_crashed_submission(tmp_path)
    broker.set_fill(intent_id, 2.0, terminal=True)
    assert not broker.cancel(intent_id)
    manager = OrderManager()
    manager.configure_intent_store(store)

    first = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=broker.lookup,
    )
    second = manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=broker.lookup,
    )

    recovered = store.get_intent(intent_id)
    assert recovered is not None
    assert recovered.status == "FILLED"
    assert first["marked_failed"] == second["marked_failed"] == 0
    fills = store.list_fills(intent_id)
    assert len(fills) == 1
    assert fills[0].fill_qty == 2.0
    assert broker.count() == 1


def test_partial_fill_and_out_of_order_updates_after_disconnect(tmp_path: Path) -> None:
    store, broker, intent_id = _start_crashed_submission(tmp_path)
    broker.set_fill(intent_id, 1.0, terminal=False)
    manager = OrderManager()
    manager.configure_intent_store(store)

    manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_client_order_id_fn=broker.lookup,
    )
    partially_filled = store.get_intent(intent_id)
    assert partially_filled is not None
    assert partially_filled.status == "PARTIALLY_FILLED"
    broker.set_fill(intent_id, 2.0, terminal=True)
    manager.reconcile_open_intents(
        broker_orders=[],
        get_order_by_id_fn=lambda _order_id: broker.lookup(intent_id),
    )
    manager.sync_external_order_state(
        intent_id=intent_id,
        order_id=f"broker-{intent_id}",
        client_order_id=intent_id,
        status="partially_filled",
        filled_qty=1.0,
        fill_price=100.25,
    )

    completed = store.get_intent(intent_id)
    assert completed is not None
    assert completed.status == "FILLED"
    assert sum(fill.fill_qty for fill in store.list_fills(intent_id)) == 2.0
    assert broker.count() == 1
