import pandas as pd

from ai_trading.trade_logic import (
    should_enter_trade,
    compute_order_price,
    extract_price,
)
from ai_trading.capital_scaling import drawdown_adjusted_kelly


def test_should_enter_trade_basic():
    assert should_enter_trade(
        [100, 105], {"signal_strength": 0.8}, {"max_risk": 0.02}
    )


def test_extract_price_generic():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    assert extract_price(df) == 2.0
    assert extract_price({"close": 3.0}) == 3.0
    assert extract_price([4.0, 5.0]) == 5.0


def test_compute_order_price_slippage():
    price = compute_order_price({"close": 10})
    assert price > 0
