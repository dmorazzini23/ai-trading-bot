import ai_trading.risk.engine as risk_engine  # AI-AGENT-REF: normalized import


def test_calculate_position_size_zero_cash():
    size = risk_engine.calculate_position_size(0, 100)
    assert size == 0


def test_calculate_position_size_negative_input():
    size = risk_engine.calculate_position_size(-1000, 100)
    assert size == 0






def test_can_trade_limits():
    eng = risk_engine.RiskEngine()
    eng.max_trades = 2
    eng.current_trades = 2
    assert not risk_engine.can_trade(eng)
    eng.current_trades = 0


