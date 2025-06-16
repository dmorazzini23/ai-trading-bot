import pytest
import risk_engine


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


def test_hard_stop_blocks_trading(monkeypatch):
    monkeypatch.setattr(risk_engine, "HARD_STOP", True)
    assert not risk_engine.can_trade()
    monkeypatch.setattr(risk_engine, "HARD_STOP", False)


def test_can_trade_limits(monkeypatch):
    monkeypatch.setattr(risk_engine, "MAX_TRADES", 2)
    monkeypatch.setattr(risk_engine, "CURRENT_TRADES", 2)
    assert not risk_engine.can_trade()
    monkeypatch.setattr(risk_engine, "CURRENT_TRADES", 0)


def test_register_and_position_size(monkeypatch):
    monkeypatch.setattr(risk_engine, "CURRENT_TRADES", 0)
    monkeypatch.setattr(risk_engine, "MAX_TRADES", 10)
    res = risk_engine.register_trade(100)
    assert res is not None
