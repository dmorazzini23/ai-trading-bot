from ai_trading.execution.simulator import SlippageModel, FillSimulator
from ai_trading.core.enums import OrderSide, OrderType


def test_update_market_conditions_zero_liquidity_does_not_raise():
    s = SlippageModel(seed=11)
    # Liquidity=0 would normally zero-divide; ensure our narrowed catch handles it.
    s.update_market_conditions(volatility=0.5, liquidity=0.0)
    assert True  # no exception


def test_slippage_uses_absolute_quantity_for_size_impact():
    positive = SlippageModel(seed=11).calculate_slippage(
        "TEST",
        OrderSide.BUY,
        quantity=100,
        price=25.0,
        order_type=OrderType.MARKET,
    )
    negative = SlippageModel(seed=11).calculate_slippage(
        "TEST",
        OrderSide.BUY,
        quantity=-100,
        price=25.0,
        order_type=OrderType.MARKET,
    )

    assert negative == positive


def test_simulate_fill_normalizes_signed_quantity():
    fs = FillSimulator(slippage_model=SlippageModel(seed=11), seed=11)
    out = fs.simulate_fill(
        symbol="TEST",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=-100,
        price=25.0,
    )

    assert out["filled"] is True
    assert 0 < out["fill_quantity"] <= 100
    assert out["partial_fills"][0]["quantity"] > 0


def test_simulate_fill_negative_price_returns_fallback():
    class FaultySlippageModel(SlippageModel):
        def calculate_slippage(self, *args, **kwargs):  # type: ignore[override]
            raise ValueError("math error")

    fs = FillSimulator(slippage_model=FaultySlippageModel(seed=11), seed=11)
    out = fs.simulate_fill(
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        price=-1.0,
    )
    assert out["filled"] is False
    assert "Simulation error" in out.get("rejection_reason", "")
