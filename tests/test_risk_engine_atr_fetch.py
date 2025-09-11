import os
import types
import logging

import pytest

from ai_trading.risk.engine import RiskEngine

os.environ.setdefault("PYTEST_RUNNING", "1")


class DummyBar:
    def __init__(self, h: float, l: float, c: float) -> None:
        self.h = h
        self.l = l
        self.c = c


def _make_valid_client():
    def get_bars(symbol: str, limit: int):
        return [DummyBar(2.0, 1.0, 1.5) for _ in range(limit)]

    return types.SimpleNamespace(get_bars=get_bars)


def test_get_atr_data_with_valid_client():
    eng = RiskEngine()
    eng.ctx = types.SimpleNamespace(data_client=_make_valid_client())
    atr = eng._get_atr_data("AAPL", lookback=14)
    assert atr == 1.0


def test_get_atr_data_missing_get_bars(caplog: pytest.LogCaptureFixture):
    eng = RiskEngine()
    eng.ctx = types.SimpleNamespace(data_client=object())
    caplog.set_level(logging.WARNING)
    atr = eng._get_atr_data("AAPL", lookback=14)
    assert atr is None
    assert "missing get_bars" in caplog.text.lower()
