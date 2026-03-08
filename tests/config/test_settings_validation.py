from datetime import timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError

from ai_trading.settings import Settings


def _new_settings(**kwargs: Any) -> Settings:
    return cast(Settings, Settings(**kwargs))


def test_risk_parameters_validated():
    """Risk ratios must be within (0, 1]."""
    with pytest.raises(
        ValueError, match=r"capital_cap must be in \(0, 1], got 1.5"
    ):
        _new_settings(capital_cap=1.5)
    with pytest.raises(
        ValueError, match=r"dollar_risk_limit must be in \(0, 1], got 0"
    ):
        _new_settings(dollar_risk_limit=0)


def test_max_position_size_positive():
    with pytest.raises(ValueError, match="max_position_size must be positive"):
        _new_settings(AI_TRADING_SIGNAL_MAX_POSITION_SIZE=-10)
    s = _new_settings(AI_TRADING_SIGNAL_MAX_POSITION_SIZE=1000)
    assert s.max_position_size == 1000


def test_missing_risk_parameter():
    with pytest.raises(ValidationError, match="Input should be a valid number"):
        _new_settings(capital_cap=None)


def test_computed_fields(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")
    s = _new_settings(ALPACA_SECRET_KEY="secret", TRADE_COOLDOWN_MIN=30)
    assert s.alpaca_secret_key_plain == "secret"
    assert s.trade_cooldown == timedelta(minutes=30)
