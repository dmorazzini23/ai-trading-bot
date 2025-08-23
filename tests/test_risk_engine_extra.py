import ai_trading.risk.engine as risk_engine  # AI-AGENT-REF: normalized import


def test_calculate_position_size_zero_cash():
    size = risk_engine.calculate_position_size(0, 100)
    assert size == 0


def test_calculate_position_size_negative_input():
    size = risk_engine.calculate_position_size(-1000, 100)
    assert size == 0


def test_check_max_drawdown_triggers_stop():
    state = {"max_drawdown": 0.5, "current_drawdown": 0.6}
    assert risk_engine.check_max_drawdown(state)


def test_check_max_drawdown_ok():
    state = {"max_drawdown": 0.5, "current_drawdown": 0.1}
    assert not risk_engine.check_max_drawdown(state)


def test_hard_stop_blocks_trading():
    eng = risk_engine.RiskEngine()
    eng.hard_stop = True
    assert not risk_engine.can_trade(eng)


def test_can_trade_limits():
    eng = risk_engine.RiskEngine()
    eng.max_trades = 2
    eng.current_trades = 2
    assert not risk_engine.can_trade(eng)
    eng.current_trades = 0


def test_register_and_position_size():
    eng = risk_engine.RiskEngine()
    eng.current_trades = 0
    eng.max_trades = 10
    res = risk_engine.register_trade(eng, 100)
    assert res is not None
