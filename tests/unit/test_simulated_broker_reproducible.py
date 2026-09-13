from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.execution.simulated_broker import SimulatedBroker


def test_day_order_expires_at_early_close_before_later_fill():
    broker = SimulatedBroker(fill_probability=1, partial_fill_probability=0)
    start = datetime(2024, 11, 29, 17, tzinfo=UTC)
    order = broker.submit_order({'symbol': 'AAPL', 'side': 'buy', 'qty': 2,
                                'type': 'limit', 'limit_price': 100,
                                'time_in_force': 'day'}, timestamp=start)
    assert order['expires_at'] == '2024-11-29T18:00:00+00:00'
    assert not broker.process_until(now=start + timedelta(minutes=5), market_price_by_symbol={'AAPL': 110})
    events = broker.process_until(now=start + timedelta(hours=1), market_price_by_symbol={'AAPL': 90})
    assert [e['event_type'] for e in events] == ['order_expired']
    assert not broker.process_until(now=start + timedelta(days=3), market_price_by_symbol={'AAPL': 90})
    assert broker.get_order(order['id'])['status'] == 'expired'
    assert broker.cancel_order(order['id']) is False


def test_unfilled_and_partial_orders_expire_without_losing_fills():
    for probability, partial in [(0, 0), (1, 1)]:
        broker = SimulatedBroker(fill_probability=probability, partial_fill_probability=partial)
        start = datetime(2024, 1, 2, 15, tzinfo=UTC)
        order = broker.submit_order({'symbol': 'AAPL', 'side': 'buy', 'qty': 100,
                                    'type': 'market', 'time_in_force': 'day'}, timestamp=start)
        broker.process_until(now=start + timedelta(minutes=5), market_price_by_symbol={'AAPL': 100})
        filled = broker.get_order(order['id'])['filled_qty']
        events = broker.process_until(now=start + timedelta(days=1), market_price_by_symbol={})
        assert len(events) == 1 and events[0]['event_type'] == 'order_expired'
        assert events[0]['filled_qty'] == filled


def test_explicit_gtc_and_unsupported_time_in_force():
    broker = SimulatedBroker(fill_probability=1, partial_fill_probability=0)
    start = datetime(2024, 1, 2, 15, tzinfo=UTC)
    order = broker.submit_order({'symbol': 'AAPL', 'side': 'buy', 'qty': 1,
                                'type': 'market', 'time_in_force': 'gtc'}, timestamp=start)
    assert order['expires_at'] is None
    assert broker.process_until(now=start + timedelta(days=1), market_price_by_symbol={'AAPL': 100})[0]['event_type'] == 'fill'
    with pytest.raises(ValueError, match='time in force'):
        broker.submit_order({'symbol': 'AAPL', 'qty': 1, 'time_in_force': 'fok'})


@pytest.mark.parametrize('fill_probability,partial', [(1, 0), (1, 1), (0, 0)])
def test_ioc_is_one_immediate_attempt_and_cancels_remainder(fill_probability, partial):
    broker = SimulatedBroker(fill_probability=fill_probability, partial_fill_probability=partial,
                             cancel_reject_probability=1)
    now = datetime(2024, 1, 2, 15, tzinfo=UTC)
    order = broker.submit_order({'symbol':'AAPL','side':'buy','qty':100,'type':'market',
                                 'time_in_force':'ioc'}, timestamp=now)
    events = broker.process_until(now=now,market_price_by_symbol={'AAPL':100})
    filled = broker.get_order(order['id'])['filled_qty']
    assert (filled == 100) if fill_probability and not partial else (filled < 100)
    assert broker.get_order(order['id'])['status'] == ('filled' if filled == 100 else 'canceled')
    canceled = [e for e in events if e['event_type']=='order_canceled']
    assert len(canceled) == int(filled < 100)
    if canceled:
        assert canceled[0]['remaining_qty'] == 100-filled
    assert all(e['ts']==now.isoformat() for e in events)
    assert not broker.process_until(now=now+timedelta(days=1),market_price_by_symbol={'AAPL':90})


@pytest.mark.parametrize('quote,delay', [(None, 0), (110, 0), (90, 60)])
def test_ioc_never_waits_for_a_later_or_marketable_quote(quote, delay):
    broker = SimulatedBroker(fill_probability=1,partial_fill_probability=0)
    now = datetime(2024,1,2,15,tzinfo=UTC)
    order = broker.submit_order({'symbol':'AAPL','side':'buy','qty':1,'type':'limit',
                                 'limit_price':100,'time_in_force':'ioc'},timestamp=now)
    events = broker.process_until(now=now+timedelta(seconds=delay),
                                  market_price_by_symbol={} if quote is None else {'AAPL':quote})
    assert [e['event_type'] for e in events] == ['order_canceled']
    assert broker.get_order(order['id'])['filled_qty']==0


def test_event_loop_accounts_ioc_fill_before_next_cap_check():
    from ai_trading.replay.event_loop import ReplayEventLoop
    broker = SimulatedBroker(fill_probability=1,partial_fill_probability=0)
    bars=[{'symbol':'AAPL','ts':'2024-01-02T15:00:00Z','close':100} for _ in range(2)]
    calls=iter(['one','two'])
    def strategy(bar):
        return {'side':'buy','qty':1,'type':'market','price':100,
                'time_in_force':'ioc','client_order_id':next(calls)}
    result=ReplayEventLoop(strategy=strategy,broker=broker,max_symbol_notional=100,
                           max_gross_notional=100).run(bars)
    assert len(result['orders'])==1
    assert len([e for e in result['events'] if e['event_type']=='fill'])==1


def test_markout_exclusions_reconcile_all_fill_events():
    from ai_trading.core.bot_engine import _replay_summary_metrics
    start = datetime(2024, 1, 2, 15, tzinfo=UTC)
    base = {'event_type': 'fill', 'side': 'buy', 'fill_price': 100,
            'fill_qty': 1, 'ts': start.isoformat()}
    events = [{**base, 'symbol': 'OK', 'order_id': 'a'},
              {**base, 'symbol': 'NONE', 'order_id': 'b'},
              {**base, 'symbol': 'LATE', 'order_id': 'c'},
              {**base, 'symbol': 'BAD', 'order_id': 'd', 'fill_price': 'bad'}]
    summary = _replay_summary_metrics({'orders': [], 'events': events}, market_rows=[
        {'symbol': 'OK', 'ts': (start + timedelta(minutes=5)).isoformat(), 'close': 101},
        {'symbol': 'LATE', 'ts': (start + timedelta(days=2)).isoformat(), 'close': 101}])
    assert summary['sample_count'] == 1
    assert summary['fill_events_seen'] == 4
    assert summary['markout_exclusion_counts'] == {
        'no_subsequent_price': 1, 'markout_horizon_exceeded': 1, 'invalid_fill_price': 1}
    assert {x['order_id'] for x in summary['markout_exclusions']} == {'b', 'c', 'd'}


@pytest.mark.parametrize('kind', ['not_submitted', 'stop', 'stop_limit', '', 'bogus'])
def test_unsupported_type_rejected_without_order_or_fill(kind):
    broker = SimulatedBroker(fill_probability=1)
    with pytest.raises(ValueError, match='unsupported replay order type'):
        broker.submit_order({'symbol': 'AAPL', 'side': 'buy', 'qty': 1, 'type': kind})
    assert broker.list_orders() == []
    assert broker.process_until(now=datetime(2024, 1, 2, tzinfo=UTC),
                                market_price_by_symbol={'AAPL': 110}) == []


@pytest.mark.parametrize("price", [None, 0, -1, float("nan"), float("inf"), "bad"])
def test_market_order_waits_for_valid_observed_price(price):
    broker = SimulatedBroker(seed=42, fill_probability=1, partial_fill_probability=0, min_fill_delay_ms=0, max_fill_delay_ms=0)
    start = datetime(2026, 9, 4, 15, tzinfo=UTC)
    order = broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 1, "type": "market", "price": 999, "client_order_id": "market"}, timestamp=start)
    quotes = {} if price is None else {"AAPL": price}
    assert broker.process_until(now=start, market_price_by_symbol=quotes) == []
    assert broker.get_order(order["id"])["filled_qty"] == 0
    later = start + timedelta(seconds=1)
    fills = broker.process_until(now=later, market_price_by_symbol={"AAPL": 200})
    assert len(fills) == 1
    assert 199 < fills[0]["fill_price"] < 202
    assert fills[0]["ts"] == later.isoformat()
    assert broker.process_until(now=later, market_price_by_symbol={"AAPL": 200}) == []


def test_replay_trailing_drain_does_not_invent_a_market_fill():
    from ai_trading.replay.event_loop import ReplayEventLoop

    replay = ReplayEventLoop(strategy=lambda bar: {"symbol": "AAPL", "side": "buy", "qty": 1, "type": "market", "price": 100, "client_order_id": "trailing"}, broker=SimulatedBroker(fill_probability=1, partial_fill_probability=0)).run([{"symbol": "AAPL", "ts": "2026-09-04T15:00:00Z", "close": 100}])
    assert len(replay["orders"]) == 1
    assert replay["events"] == []


def _run(seed: int) -> tuple[dict, list[dict], dict | None]:
    broker = SimulatedBroker(
        seed=seed,
        fill_probability=1.0,
        partial_fill_probability=0.5,
    )
    submitted_at = datetime(2026, 2, 18, 15, 0, tzinfo=UTC)
    order = broker.submit_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "repro-test",
        },
        timestamp=submitted_at,
        spread_bps=7.5,
        volatility_pct=0.015,
    )
    events = broker.process_until(
        now=submitted_at + timedelta(minutes=10),
        market_price_by_symbol={"AAPL": 189.0},
    )
    snapshot = broker.get_order(order["id"])
    return order, events, snapshot


def test_simulated_broker_reproducible_for_same_seed() -> None:
    first = _run(seed=123)
    second = _run(seed=123)
    assert first == second


def test_simulated_broker_varies_for_different_seed() -> None:
    first = _run(seed=123)
    second = _run(seed=124)
    assert first != second


def test_simulated_broker_cancel_reject_probability() -> None:
    broker = SimulatedBroker(
        seed=321,
        fill_probability=1.0,
        partial_fill_probability=0.0,
        cancel_reject_probability=1.0,
    )
    submitted_at = datetime(2026, 2, 18, 15, 0, tzinfo=UTC)
    order = broker.submit_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 1,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "cancel-reject",
        },
        timestamp=submitted_at,
    )
    cancelled = broker.cancel_order(order["id"], timestamp=submitted_at)
    assert cancelled is False
    events = broker.process_until(
        now=submitted_at + timedelta(seconds=1),
        market_price_by_symbol={"AAPL": 190.0},
    )
    assert any(event.get("event_type") == "cancel_rejected" for event in events)


def test_simulated_broker_emits_single_fill_event_per_schedule() -> None:
    broker = SimulatedBroker(
        seed=11,
        fill_probability=1.0,
        partial_fill_probability=0.0,
        min_fill_delay_ms=0,
        max_fill_delay_ms=0,
    )
    submitted_at = datetime(2026, 2, 18, 15, 0, tzinfo=UTC)
    order = broker.submit_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 2,
            "type": "limit",
            "limit_price": 190.0,
            "client_order_id": "single-fill",
        },
        timestamp=submitted_at,
    )
    events = broker.process_until(
        now=submitted_at + timedelta(seconds=1),
        market_price_by_symbol={"AAPL": 190.0},
    )
    fills = [event for event in events if event.get("event_type") == "fill"]
    assert len(fills) == 1
    assert fills[0]["order_id"] == order["id"]


def test_limit_waits_for_market_and_never_fills_through_limit():
    for side, unreachable in [("buy", 101.0), ("sell", 99.0)]:
        broker = SimulatedBroker(fill_probability=1, partial_fill_probability=0, min_fill_delay_ms=0, max_fill_delay_ms=0)
        start = datetime(2026, 2, 18, 15, tzinfo=UTC)
        broker.submit_order({"symbol": "AAPL", "side": side, "qty": 2, "type": "limit", "limit_price": 100, "client_order_id": side}, timestamp=start)
        assert broker.process_until(now=start, market_price_by_symbol={}) == []
        assert broker.process_until(now=start, market_price_by_symbol={"AAPL": unreachable}) == []
        later = start + timedelta(minutes=1)
        fills = broker.process_until(now=later, market_price_by_symbol={"AAPL": 100})
        assert len(fills) == 1
        assert fills[0]["fill_price"] <= 100 if side == "buy" else fills[0]["fill_price"] >= 100
        assert fills[0]["ts"] == later.isoformat()


def test_unrelated_removed_order_does_not_change_retained_order_randomness():
    results = []
    start = datetime(2026, 2, 18, 15, tzinfo=UTC)
    for extra in [True, False]:
        broker = SimulatedBroker(seed=42, fill_probability=1)
        if extra:
            broker.submit_order({"symbol": "MSFT", "side": "buy", "qty": 5, "type": "market", "price": 200, "client_order_id": "removed"}, timestamp=start)
        broker.submit_order({"symbol": "AAPL", "side": "buy", "qty": 10, "type": "market", "price": 100, "client_order_id": "retained"}, timestamp=start)
        events = broker.process_until(now=start + timedelta(minutes=1), market_price_by_symbol={"AAPL": 100, "MSFT": 200})
        retained = next(row for row in events if row["client_order_id"] == "retained")
        results.append({key: retained[key] for key in ["fill_qty", "fill_price", "ts"]})
    assert results[0] == results[1]


def test_simulated_fee_contract_survives_replay_summary():
    from ai_trading.core.bot_engine import _replay_summary_metrics
    from ai_trading.replay.event_loop import ReplayEventLoop
    from ai_trading.tools.paper_evidence_review import _fee_total

    bars = [{"ts": f"2026-08-03T15:{minute:02d}:00Z", "symbol": "AAPL", "close": 100} for minute in (0, 5, 10)]
    def strategy(bar):
        return {"symbol": "AAPL", "side": "buy", "qty": 2, "type": "market", "price": 100, "client_order_id": "fee-test"} if bar["ts"].endswith("00:00Z") else None
    broker = SimulatedBroker(fee_bps=1, fill_probability=1, partial_fill_probability=0)
    result = ReplayEventLoop(strategy=strategy, broker=broker).run(bars)
    summary = _replay_summary_metrics(result, market_rows=bars)
    rows = summary["markout_observations"]
    assert len(rows) == 1
    from json import dumps, loads
    saved = loads(dumps(rows[0]))
    assert saved['decision_ts'] == '2026-08-03T15:00:00+00:00'
    assert saved['reference_ts'] == saved['decision_ts']
    assert saved['fill_ts'] == '2026-08-03T15:05:00+00:00'
    assert saved['markout_ts'] == '2026-08-03T15:10:00+00:00'
    assert saved['decision_to_fill_seconds'] == 300
    assert saved['markout_horizon_seconds'] == 300
    assert saved['fill_assumptions']['spread_bps'] == 8
    assert saved['fill_assumptions']['volatility_pct'] == .01
    assert saved['fill_assumptions']['fee_bps'] == 1
    assert saved['fill_assumptions']['market_price'] == 100
    assert _fee_total(rows, simulated=True) == pytest.approx(rows[0]["fill_qty"] * rows[0]["fill_price"] / 10000)
    gross_markout = (rows[0]["markout_price"] / rows[0]["fill_price"] - 1) * 10000
    assert summary["net_edge_bps"] == pytest.approx(gross_markout - 1)
    with pytest.raises(ValueError, match="fee_bps"):
        SimulatedBroker(fee_bps=float("nan"))
