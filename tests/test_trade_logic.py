from ai_trading.trade_logic import should_enter_trade
from ai_trading.capital_scaling import drawdown_adjusted_kelly


def test_should_enter_trade_basic():
    assert should_enter_trade(
        [100, 105], {"signal_strength": 0.8}, {"max_risk": 0.02}
    )
