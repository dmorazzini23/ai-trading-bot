from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.analytics.tca import (
    ExecutionBenchmark,
    FillSummary,
    build_tca_record,
    implementation_shortfall_bps,
)


def test_implementation_shortfall_buy_direction() -> None:
    # Buy worse fill than decision price should be positive cost bps
    value = implementation_shortfall_bps("buy", 100.0, 101.0, fees=0.0, qty=10)
    assert round(value, 6) == 100.0


def test_implementation_shortfall_sell_direction() -> None:
    # Sell worse fill than decision price should also be positive cost bps
    value = implementation_shortfall_bps("sell", 100.0, 99.0, fees=0.0, qty=10)
    assert round(value, 6) == 100.0


def test_tca_record_includes_canonical_price_fields() -> None:
    ts = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
    record = build_tca_record(
        client_order_id="cid-1",
        symbol="AAPL",
        side="buy",
        benchmark=ExecutionBenchmark(
            arrival_price=100.0,
            mid_at_arrival=100.2,
            decision_ts=ts,
            submit_ts=ts,
            first_fill_ts=ts,
        ),
        fill=FillSummary(fill_vwap=100.5, total_qty=10, fees=0.0, status="filled"),
    )
    assert record["decision_price"] == 100.0
    assert record["submit_price_reference"] == 100.2
    assert record["fill_price"] == 100.5
