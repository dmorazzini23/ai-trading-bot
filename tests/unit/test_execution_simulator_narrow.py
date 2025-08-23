from ai_trading.execution.simulator import SlippageModel, FillSimulator
from ai_trading.core.enums import OrderSide, OrderType


def test_update_market_conditions_zero_liquidity_does_not_raise():
    s = SlippageModel()
    # Liquidity=0 would normally zero-divide; ensure our narrowed catch handles it.
    s.update_market_conditions(volatility=0.5, liquidity=0.0)
    assert True  # no exception


def test_simulate_fill_negative_price_returns_fallback():
    class FaultySlippageModel(SlippageModel):
        def calculate_slippage(self, *args, **kwargs):  # type: ignore[override]
            raise ValueError("math error")

    fs = FillSimulator(slippage_model=FaultySlippageModel())
    out = fs.simulate_fill(
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        price=-1.0,
    )
    assert out["filled"] is False
    assert "Simulation error" in out.get("rejection_reason", "")

