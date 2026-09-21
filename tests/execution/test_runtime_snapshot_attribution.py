from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from ai_trading.execution import live_trading as lt


def test_broker_snapshots_preserve_signed_inventory_and_lineage(monkeypatch):
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    store = Mock()
    engine._resolve_runtime_snapshot_store = lambda: store
    engine._runtime_snapshot_lineage = lambda: {'policy_hash': 'policy', 'model_artifact_hash': 'model', 'config_snapshot_hash': 'config'}
    engine._cycle_account = {'equity': 10000, 'buying_power': 5000}
    engine.ctx = NS(exposure_pct=.2)
    engine._emit_runtime_snapshots_from_broker_sync(open_orders=({'id': 'one'},), positions=(
        {'symbol': 'aapl', 'qty': 2, 'side': 'long', 'current_price': 100},
        {'symbol': 'amzn', 'qty': 3, 'side': 'short', 'current_price': 200},
        {'symbol': '', 'qty': 10}, {'symbol': 'MSFT', 'qty': 'invalid'},
    ), open_buy_by_symbol={'MSFT': 4}, open_sell_by_symbol={'AMZN': 1})
    calls = store.append_position_snapshot_payload.call_args_list
    assert len(calls) == 2
    assert [c.kwargs['quantity'] for c in calls] == [2., -3.]
    assert [c.kwargs['market_value'] for c in calls] == [200., -600.]
    assert all(c.kwargs['model_hash'] == 'model' for c in calls)
    risk = store.append_risk_snapshot_payload.call_args.kwargs
    assert risk['config_hash'] == 'config'
    assert risk['payload']['net_position_qty'] == -1.
    assert risk['payload']['gross_position_qty'] == 5.
    assert risk['payload']['emitted_positions'] == 2
    assert risk['payload']['open_buy_qty_total'] == 4.
    assert risk['payload']['open_sell_qty_total'] == 1.


@pytest.mark.parametrize('use_context', [False, True])
def test_dual_feed_record_keeps_execution_and_reference_evidence_separate(monkeypatch, use_context):
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine._runtime_exec_event_persistence_enabled = lambda: True
    engine._append_runtime_jsonl = Mock()
    monkeypatch.setattr(lt, 'get_execution_feed', lambda *a: 'iex')
    monkeypatch.setattr(lt, 'get_reference_feed', lambda: 'sip_delayed')
    monkeypatch.setattr(lt, 'fetch_reference_snapshot', lambda *a, **k:
                        dict(price=102., bid=101., ask=103., volume=1000))
    context = {'signal_strength': .5, 'signal_confidence': .7, 'signal_weight': .2, 'signal_side': 'buy'} if use_context else {}
    original = context.copy()
    engine._record_dual_feed_decision(symbol='aapl', side='buy', quantity=2, outcome='accepted',
        execution_price=100., execution_bid=99., execution_ask=101., execution_volume=500,
        context=context, signal=NS(side='buy', strength=.5, confidence=.7, weight=.2))
    record = engine._append_runtime_jsonl.call_args.kwargs['payload']
    assert record['execution_feed'] == 'iex'
    assert record['reference_feed'] == 'sip_delayed'
    assert record['execution_price'] == 100. and record['reference_price'] == 102.
    assert record['symbol'] == 'AAPL' and record['quantity'] == 2
    assert record['signal_strength'] == .5 and record['signal_weight'] == .2
    assert record['decision_id'] and record['decision_ts']
    assert context == original
