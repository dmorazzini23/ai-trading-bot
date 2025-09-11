import pytest

from ai_trading.execution import ExecutionEngine
from ai_trading.core.enums import OrderSide


def test_delayed_quote_slippage_flagged(monkeypatch):
    """Large move between quote and fill should trigger slippage alert."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "50")
    prices = iter([100.0, 102.0])
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.get_latest_price",
        lambda symbol: next(prices),
    )
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 50, raising=False)
    engine = ExecutionEngine()
    with pytest.raises(AssertionError):
        engine.execute_order("AAPL", OrderSide.BUY, 10)
    order = next(iter(engine.order_manager.orders.values()))
    assert order.slippage_bps > 50


def test_delayed_quote_slippage_within_threshold(monkeypatch):
    """Minor quote movement should record slippage without alert."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "50")
    prices = iter([100.0, 100.3])
    monkeypatch.setattr(
        "ai_trading.core.bot_engine.get_latest_price",
        lambda symbol: next(prices),
    )
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 50, raising=False)
    engine = ExecutionEngine()
    order_id = engine.execute_order("AAPL", OrderSide.BUY, 10)
    assert order_id is not None
    order = engine.order_manager.orders[order_id]
    assert float(order.expected_price) == pytest.approx(100.0)
    assert order.slippage_bps == pytest.approx(30.0)
    assert abs(order.slippage_bps) < 50
