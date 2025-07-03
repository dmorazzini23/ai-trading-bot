from trade_logic import should_enter_trade


def test_should_enter_trade_basic():
    assert should_enter_trade(
        [100, 105], {"signal_strength": 0.8}, {"max_risk": 0.02}
    )
