import logging
import os
import types

os.environ.setdefault("PYTEST_RUNNING", "1")
os.environ.setdefault("MAX_DRAWDOWN_THRESHOLD", "0.15")

import pandas as pd
import pytest

from ai_trading.data.providers import yfinance_provider
from ai_trading.risk.engine import RiskEngine


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


def test_get_atr_data_with_yfinance(monkeypatch: pytest.MonkeyPatch):
    class DummyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, period: str, interval: str):
            n = int(str(period).rstrip("d"))
            data = {
                "Open": [1.0] * n,
                "High": [2.0] * n,
                "Low": [1.0] * n,
                "Close": [1.5] * n,
                "Volume": [10] * n,
            }
            return pd.DataFrame(data)

    dummy_yf = types.SimpleNamespace(Ticker=lambda symbol: DummyTicker(symbol))
    monkeypatch.setattr(yfinance_provider, "get_yfinance", lambda: dummy_yf)

    provider = yfinance_provider.Provider()
    eng = RiskEngine()
    eng.ctx = types.SimpleNamespace(data_client=provider)
    atr = eng._get_atr_data("AAPL", lookback=14)
    assert atr == 1.0
