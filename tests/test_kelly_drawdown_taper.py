import pytest
from capital_scaling import drawdown_adjusted_kelly as drawdown_adjusted_kelly_alt


def test_drawdown_adjusted_kelly_basic():
    assert 0.0 <= drawdown_adjusted_kelly_alt(9000, 10000, 0.5) <= 1.0


def test_drawdown_adjusted_kelly_zero_drawdown():
    assert drawdown_adjusted_kelly_alt(10000, 10000, 0.5) > 0.0
