from datetime import timedelta

import pytest
from pydantic import ValidationError

from ai_trading.settings import Settings


def test_risk_parameters_validated():
    """Risk ratios must be within (0, 1]."""
    with pytest.raises(
        ValueError, match=r"capital_cap must be in \(0, 1], got 1.5"
    ):
        Settings(capital_cap=1.5)
    with pytest.raises(
        ValueError, match=r"dollar_risk_limit must be in \(0, 1], got 0"
    ):
        Settings(dollar_risk_limit=0)


def test_max_position_size_positive():
    with pytest.raises(ValueError, match="max_position_size must be positive"):
        Settings(MAX_POSITION_SIZE=-10)
    s = Settings(MAX_POSITION_SIZE=1000)
    assert s.max_position_size == 1000


def test_missing_risk_parameter():
    with pytest.raises(ValidationError, match="Input should be a valid number"):
        Settings(capital_cap=None)


def test_computed_fields(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")
    s = Settings(ALPACA_SECRET_KEY="secret", TRADE_COOLDOWN_MIN=30)
    assert s.alpaca_secret_key_plain == "secret"
    assert s.trade_cooldown == timedelta(minutes=30)
