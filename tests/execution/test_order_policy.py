from datetime import UTC, datetime

import pytest

from ai_trading.execution.order_policy import MarketData, OrderUrgency, SmartOrderRouter


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


def test_phase2_execution_edge_routing_is_opt_in_by_default() -> None:
    router = SmartOrderRouter()

    order = router.create_order_request(
        symbol="AAPL",
        side="buy",
        quantity=10,
        market_data=_market_data(),
        urgency=OrderUrgency.LOW,
        execution_context={
            "samples": 40,
            "fill_rate": 0.1,
            "reject_rate": 0.0,
            "mean_slippage_bps": 1.0,
        },
    )

    assert order["limit_price"] == pytest.approx(99.95)
    assert "execution_edge_routing" not in order


def test_phase2_execution_edge_routing_adapts_buy_offset_when_guardrails_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = SmartOrderRouter()
    market_data = _market_data()
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_ROUTING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MIN_SAMPLES", "10")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_FILL_RATE", "0.8")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MAX_REJECT_RATE", "0.05")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_SLIPPAGE_BPS", "5")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MAX_OFFSET_ADD_BPS", "4")

    order = router.create_order_request(
        symbol="AAPL",
        side="buy",
        quantity=10,
        market_data=market_data,
        urgency=OrderUrgency.LOW,
        execution_context={
            "samples": 40,
            "fill_rate": 0.6,
            "reject_rate": 0.02,
            "mean_slippage_bps": 2.0,
        },
    )

    assert order["limit_price"] == pytest.approx(99.982)
    edge_context = order["execution_edge_routing"]
    assert edge_context["applied"] is True
    assert edge_context["reason"] == "low_fill_rate_guarded_offset"
    assert edge_context["adaptive_offset_add_bps"] == pytest.approx(3.2)


def test_phase2_execution_edge_routing_preserves_short_side_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = SmartOrderRouter()
    market_data = _market_data()
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_ROUTING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MIN_SAMPLES", "10")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_FILL_RATE", "0.8")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MAX_OFFSET_ADD_BPS", "4")

    order = router.create_order_request(
        symbol="AAPL",
        side="sell_short",
        quantity=10,
        market_data=market_data,
        urgency=OrderUrgency.LOW,
        execution_context={
            "samples": 40,
            "fill_rate": 0.6,
            "reject_rate": 0.0,
            "mean_slippage_bps": 1.0,
        },
    )

    assert order["side"] == "sell_short"
    assert order["limit_price"] == pytest.approx(100.018)
    assert order["limit_price"] > market_data.mid
    assert order["execution_edge_routing"]["applied"] is True


def test_phase2_execution_edge_routing_treats_buy_to_cover_as_buy_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = SmartOrderRouter()
    market_data = _market_data()
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_ROUTING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MIN_SAMPLES", "10")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_FILL_RATE", "0.8")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MAX_OFFSET_ADD_BPS", "4")

    order = router.create_order_request(
        symbol="AAPL",
        side="buy_to_cover",
        quantity=10,
        market_data=market_data,
        urgency=OrderUrgency.LOW,
        execution_context={
            "samples": 40,
            "fill_rate": 0.6,
            "reject_rate": 0.0,
            "mean_slippage_bps": 1.0,
        },
    )

    assert order["side"] == "buy_to_cover"
    assert order["limit_price"] == pytest.approx(99.982)
    assert order["limit_price"] < market_data.mid
    assert order["execution_edge_routing"]["applied"] is True


def test_phase2_execution_edge_routing_reject_guard_blocks_widening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = SmartOrderRouter()
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_ROUTING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MIN_SAMPLES", "10")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_FILL_RATE", "0.8")
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MAX_REJECT_RATE", "0.05")

    order = router.create_order_request(
        symbol="AAPL",
        side="buy",
        quantity=10,
        market_data=_market_data(),
        urgency=OrderUrgency.LOW,
        execution_context={
            "samples": 40,
            "fill_rate": 0.6,
            "reject_rate": 0.12,
            "mean_slippage_bps": 1.0,
        },
    )

    assert order["limit_price"] == pytest.approx(99.95)
    assert order["execution_edge_routing"]["applied"] is False
    assert order["execution_edge_routing"]["reason"] == "reject_rate_guard"
