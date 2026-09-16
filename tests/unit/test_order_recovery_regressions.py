from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from alpaca.common.exceptions import APIError

from ai_trading.execution.engine import OrderManager
from ai_trading.execution.position_reconciler import PositionReconciler
from ai_trading.oms.intent_store import IntentStore


def make_store(path):
    store = IntentStore(path=str(path), event_dual_write_enabled=False)
    store.create_intent(intent_id='i', idempotency_key='k', symbol='AAPL', side='buy', quantity=10)
    return store


@pytest.mark.parametrize('status', ['FILLED', 'CANCELED', 'EXPIRED', 'REJECTED'])
def test_terminal_state_survives_late_callbacks(tmp_path, status):
    store = make_store(tmp_path / 'terminal.db')
    try:
        store.close_intent('i', final_status=status)
        store.record_submit_error('i', 'late timeout')
        store.mark_submitted('i', 'broker')
        store.record_fill('i', fill_qty=1, fill_price=100)
        store.close_intent('i', final_status='FAILED')
        assert store.get_intent('i').status == status
        assert not store.claim_for_submit('i')
        assert store.get_open_intents() == []
        assert len(store.list_fills('i')) == 1
    finally:
        store.close()


def test_cumulative_callbacks_across_connections_and_restart(tmp_path):
    path = tmp_path / 'fills.db'
    first = make_store(path)
    second = IntentStore(path=str(path), event_dual_write_enabled=False)
    barrier = Barrier(2)

    def callback(store):
        manager = OrderManager.__new__(OrderManager)
        manager._intent_store = store
        manager._intent_by_order_id = {}
        manager._intent_reported_fill_qty = {}
        barrier.wait(timeout=5)
        manager.sync_external_order_state(intent_id='i', status='partially_filled', filled_qty=2, fill_price=100)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(callback, [first, second]))
        assert sum(row.fill_qty for row in first.list_fills('i')) == 2
    finally:
        first.close()
        second.close()
    resumed = IntentStore(path=str(path), event_dual_write_enabled=False)
    try:
        for total in (2, 1, 5, 5):
            resumed.record_fill('i', fill_qty=total, fill_price=100, cumulative=True)
        assert [row.fill_qty for row in resumed.list_fills('i')] == [2, 3]
    finally:
        resumed.close()


@pytest.mark.parametrize('snapshot', [None, [SimpleNamespace(symbol='AAPL', qty='nan')], [SimpleNamespace(qty=1)]])
def test_invalid_snapshot_preserves_positions(snapshot):
    reconciler = PositionReconciler(SimpleNamespace(get_all_positions=lambda: snapshot))
    reconciler.update_bot_position('AAPL', 5)
    assert reconciler.force_sync_from_broker() == {'AAPL': 5}
    assert reconciler._last_broker_fetch_failed


def test_api_error_recovers_and_worker_stops(monkeypatch):
    calls = []

    def fetch():
        calls.append(1)
        if len(calls) == 1:
            raise APIError('{"code":429,"message":"rate limited"}')
        return [SimpleNamespace(symbol='AAPL', qty=5)]

    reconciler = PositionReconciler(SimpleNamespace(get_all_positions=fetch))
    reconciler.update_bot_position('AAPL', 5)
    monkeypatch.setattr(reconciler._stop_event, 'wait', lambda delay: len(calls) >= 2)
    reconciler.running = True
    reconciler._reconciliation_loop()
    assert len(calls) == 2
    assert not reconciler.running
    assert not reconciler._last_broker_fetch_failed
    assert reconciler.get_bot_positions() == {'AAPL': 5}


def test_stop_interrupts_idle_wait_and_allows_restart():
    reconciler = PositionReconciler(SimpleNamespace(get_all_positions=lambda: []))
    for _ in range(2):
        reconciler.start_periodic_reconciliation(interval=300)
        reconciler.stop_periodic_reconciliation()
        assert not reconciler.running
        assert not reconciler._reconciliation_thread.is_alive()
