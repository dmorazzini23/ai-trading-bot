from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.execution.simulated_broker import SimulatedBroker


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
    assert _fee_total(rows, simulated=True) == pytest.approx(rows[0]["fill_qty"] * rows[0]["fill_price"] / 10000)
    gross_markout = (rows[0]["markout_price"] / rows[0]["fill_price"] - 1) * 10000
    assert summary["net_edge_bps"] == pytest.approx(gross_markout - 1)
    with pytest.raises(ValueError, match="fee_bps"):
        SimulatedBroker(fee_bps=float("nan"))
