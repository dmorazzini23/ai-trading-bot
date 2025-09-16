import logging
import os
import types

os.environ.setdefault("PYTEST_RUNNING", "1")
os.environ.setdefault("MAX_DRAWDOWN_THRESHOLD", "0.15")

import pandas as pd
import pytest

from ai_trading.config.settings import get_settings
from ai_trading.data.providers import yfinance_provider
from ai_trading.risk.engine import RiskEngine


class _StockBarsClient:
    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars
        self.requests: list[object] = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        return self._bars


def _make_df(rows: int = 30) -> pd.DataFrame:
    data = {
        "open": [1.0] * rows,
        "high": [2.0] * rows,
        "low": [1.0] * rows,
        "close": [1.5] * rows,
    }
    return pd.DataFrame(data)


def test_get_atr_data_with_stock_bars_client(caplog: pytest.LogCaptureFixture):
    eng = RiskEngine()
    client = _StockBarsClient(_make_df())
    eng.ctx = types.SimpleNamespace(data_client=client)
    caplog.set_level(logging.WARNING)
    atr = eng._get_atr_data("AAPL", lookback=14)
    assert atr == 1.0
    assert "missing stock bars fetch" not in caplog.text.lower()
    assert client.requests, "expected get_stock_bars to be invoked"
    request = client.requests[-1]
    assert getattr(request, "limit", None) == 24
    timeframe = getattr(request, "timeframe", None)
    if timeframe is not None:
        assert "day" in str(timeframe).lower()
    expected_feed = getattr(get_settings(), "alpaca_data_feed", None)
    if expected_feed:
        assert getattr(request, "feed", None) == expected_feed


def test_get_atr_data_missing_get_bars_no_data(caplog: pytest.LogCaptureFixture):
    eng = RiskEngine()
    eng.ctx = types.SimpleNamespace(data_client=object())
    caplog.set_level(logging.WARNING)
    atr = eng._get_atr_data("AAPL", lookback=14)
    assert atr is None
    assert "missing stock bars fetch" in caplog.text.lower()


def test_get_atr_data_uses_ctx_minute_data():
    eng = RiskEngine()
    df = _make_df()
    eng.ctx = types.SimpleNamespace(data_client=object(), minute_data={"AAPL": df})
    atr = eng._get_atr_data("AAPL", lookback=14)
    assert atr == 1.0


def test_get_atr_data_uses_ctx_daily_data():
    eng = RiskEngine()
    df = _make_df()
    eng.ctx = types.SimpleNamespace(data_client=object(), daily_data={"AAPL": df})
    atr = eng._get_atr_data("AAPL", lookback=14)
    assert atr == 1.0


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
