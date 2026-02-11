from datetime import UTC, datetime

from ai_trading.execution.order_policy import MarketData, SmartOrderRouter


def _market_data() -> MarketData:
    return MarketData(
        symbol="AAPL",
        bid=99.9,
        ask=100.1,
        mid=100.0,
        spread_bps=20.0,
        volume_ratio=1.0,
    )


def test_should_cancel_and_retry_handles_string_quantities() -> None:
    router = SmartOrderRouter()
    router._active_orders["ord-1"] = {
        "symbol": "AAPL",
        "created_at": datetime.now(UTC),
        "limit_price": 100.0,
    }

    should_cancel, reason = router.should_cancel_and_retry(
        "ord-1",
        {"filled_quantity": "0", "quantity": "10"},
        _market_data(),
    )

    assert should_cancel is False
    assert reason == "Order should continue"


def test_should_cancel_and_retry_handles_zero_total_quantity() -> None:
    router = SmartOrderRouter()
    router._active_orders["ord-2"] = {
        "symbol": "AAPL",
        "created_at": datetime.now(UTC),
        "limit_price": 100.0,
    }

    should_cancel, reason = router.should_cancel_and_retry(
        "ord-2",
        {"filled_quantity": 0, "quantity": 0},
        _market_data(),
    )

    assert should_cancel is False
    assert reason == "Order should continue"
