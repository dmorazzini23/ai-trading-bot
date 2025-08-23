
import ai_trading.risk.engine as risk_engine  # AI-AGENT-REF: normalized import
import numpy as np
import pytest
from ai_trading.strategies import TradeSignal


def test_can_trade_invalid_type(caplog):
    """can_trade rejects non-TradeSignal objects."""
    eng = risk_engine.RiskEngine()
    caplog.set_level('ERROR')
    assert not eng.can_trade(object())
    assert "invalid signal" in caplog.text


def test_register_fill_invalid(caplog):
    """register_fill ignores invalid types."""
    eng = risk_engine.RiskEngine()
    caplog.set_level('ERROR')
    eng.register_fill(object())
    assert "invalid signal" in caplog.text


def test_check_max_drawdown_exception(monkeypatch):
    """API errors lead to False return."""
    class API:
        def get_account(self):
            raise RuntimeError('oops')
    eng = risk_engine.RiskEngine()
    assert not eng.check_max_drawdown(API())


def test_position_size_invalid_signal():
    """Invalid signal results in zero position."""
    eng = risk_engine.RiskEngine()
    assert eng.position_size(object(), 100, 10) == 0


def test_position_size_division_error():
    """Errors during quantity calc return zero."""
    eng = risk_engine.RiskEngine()
    sig = TradeSignal(symbol='A', side='buy', confidence=1.0, strategy='s')
    qty = eng.position_size(sig, cash=100, price=float('nan'))
    assert qty == 0


def test_apply_weight_limits():
    """Weight adjustments respect caps."""
    eng = risk_engine.RiskEngine()
    eng.asset_limits['equity'] = 0.5
    eng.strategy_limits['s'] = 0.3
    eng.exposure['equity'] = 0.4
    sig = TradeSignal(symbol='A', side='buy', confidence=1.0, strategy='s', weight=1.0)
    w = eng._apply_weight_limits(sig)
    assert round(w, 1) == 0.1


def test_compute_volatility_error(monkeypatch):
    """Exceptions in std calculation are handled."""
    eng = risk_engine.RiskEngine()
    monkeypatch.setattr(risk_engine.np, 'std', lambda arr: (_ for _ in ()).throw(ValueError('bad')))
    res = eng.compute_volatility(np.array([1.0]))
    assert res['volatility'] == 0.0


def test_compute_volatility_nan(caplog):
    eng = risk_engine.RiskEngine()
    caplog.set_level("ERROR")
    arr = np.array([1.0, np.nan])
    res = eng.compute_volatility(arr)
    assert res["volatility"] == 0.0
    assert "invalid values" in caplog.text


def test_calculate_position_size_invalid_args():
    """Invalid argument patterns raise TypeError."""
    with pytest.raises(TypeError):
        risk_engine.calculate_position_size()


def test_register_trade_blocked():
    """register_trade returns None when trading not allowed."""
    eng = risk_engine.RiskEngine()
    eng.max_trades = 1
    eng.current_trades = 1
    assert risk_engine.register_trade(eng, 1) is None
