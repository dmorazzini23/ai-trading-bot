from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.execution.simulated_broker import SimulatedBroker


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
        market_price_by_symbol={"AAPL": 191.0},
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
