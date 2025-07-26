import pytest
from ai_trading.capital_scaling import volatility_parity_position


def test_volatility_parity_position_basic():
    assert volatility_parity_position(0.2, 0.5) > 0.0


def test_volatility_parity_zero_vol():
    assert volatility_parity_position(0.0, 0.5) == 0.01
