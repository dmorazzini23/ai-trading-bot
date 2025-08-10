import datetime
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
mods = [
    "alpaca",
    "alpaca.data.historical",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
    "alpaca_trade_api.rest",
    "alpaca.common.exceptions",
    "dotenv",
    "finnhub",
]
for m in mods:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules.setdefault("alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
sys.modules["dotenv"] = types.ModuleType("dotenv")
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules["dotenv"].dotenv_values = lambda *a, **k: {}


class _FakeREST:
    def __init__(self, *a, **k):
        pass

    def get_bars(self, *a, **k):
        return types.SimpleNamespace(df=pd.DataFrame())


sys.modules["alpaca_trade_api"].REST = _FakeREST
sys.modules["alpaca_trade_api"].APIError = Exception
sys.modules["alpaca_trade_api.rest"].REST = _FakeREST
sys.modules["alpaca_trade_api.rest"].APIError = Exception
sys.modules["alpaca_trade_api.rest"].TimeFrame = object
sys.modules["alpaca.common.exceptions"].APIError = Exception


class _DummyHist:
    def __init__(self, *a, **k):
        pass

    def get_stock_bars(self, *a, **k):
        import pandas as pd

        return types.SimpleNamespace(df=pd.DataFrame())


sys.modules["alpaca.data.historical"].StockHistoricalDataClient = _DummyHist


class _DummyRequest:
    def __init__(self, *a, **k):
        for key, val in k.items():
            setattr(self, key, val)


sys.modules["alpaca.data.requests"].StockBarsRequest = _DummyRequest
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = _DummyRequest
sys.modules["alpaca.data.timeframe"].TimeFrame = object
sys.modules["alpaca.data.timeframe"].TimeFrameUnit = object


class _DummyFinnhub:
    def __init__(self, *a, **k):
        pass


sys.modules["finnhub"].Client = _DummyFinnhub
class _DummyFinnhubException(Exception):
    def __init__(self, status_code=0):
        super().__init__("finnhub error")
        self.status_code = status_code

sys.modules["finnhub"].FinnhubAPIException = _DummyFinnhubException

from ai_trading import data_fetcher


class FakeBars:
    def __init__(self, df: pd.DataFrame):
        self.df = df


def test_get_minute_df(monkeypatch):
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
        index=[pd.Timestamp("2023-01-01T09:30")],
    )

    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: df.reset_index().rename(columns={"index": "timestamp"}))
    monkeypatch.setattr(data_fetcher, "is_market_open", lambda: True)
    result = data_fetcher.get_minute_df("AAPL", datetime.date(2023, 1, 1), datetime.date(2023, 1, 2))
    assert not result.empty


def test_subscription_error_logged(monkeypatch, caplog):
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
        index=[pd.Timestamp("2023-01-01T09:30")],
    )

    class DummyClient:
        def get_stock_bars(self, req):
            if getattr(req, "feed", None) == "iex":
                return FakeBars(df)
            raise data_fetcher.APIError("subscription does not permit querying recent SIP data")

    monkeypatch.setattr(data_fetcher, "client", DummyClient())
    monkeypatch.setattr(data_fetcher, "TimeFrame", types.SimpleNamespace(Minute="1Min"))
    def fake_yf(sym):
        idx = pd.date_range(start="2023-01-01 09:30", periods=5, freq="1min", tz="UTC")
        return pd.DataFrame(
            {"timestamp": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
        )

    monkeypatch.setattr(data_fetcher, "fetch_minute_yfinance", fake_yf)
    monkeypatch.setattr(data_fetcher.fh_fetcher, "fetch", lambda *a, **k: fake_yf("AAPL"))

    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2023-01-02", tz="UTC")
    messages = []
    monkeypatch.setattr(data_fetcher.logger, "critical", lambda msg, *a, **k: messages.append(msg))
    data_fetcher.get_minute_df("AAPL", start, end)
    assert messages == []


def test_default_feed_constant():
    assert data_fetcher._DEFAULT_FEED == "iex"


def test_fetch_bars_retry_invalid_feed(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, status, text, data=None):
            self.status_code = status
            self.text = text
            self._data = data or {}

        def json(self):
            return self._data

    def fake_get(url, params=None, headers=None, timeout=10):
        calls.append(params["feed"])
        if len(calls) == 1:
            return Resp(400, "invalid feed")
        return Resp(200, "", {"bars": [{"t": "2023-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]})

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)

    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = start + pd.Timedelta(minutes=1)
    df = data_fetcher._fetch_bars("AAPL", start, end, "1Min", "iex")
    assert calls == ["iex", "sip"]
    assert not df.empty


def test_finnhub_403_yfinance(monkeypatch):
    def raise_fetch(*a, **k):
        raise data_fetcher.DataFetchException("AAPL", "alpaca", "", "err")

    def raise_finnhub(*a, **k):
        raise data_fetcher.FinnhubAPIException(status_code=403)

    called = []

    def fake_yf(symbol):
        called.append(symbol)
        return pd.DataFrame(
            {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]},
            index=[pd.Timestamp("2023-01-01", tz="UTC")],
        )

    monkeypatch.setattr(data_fetcher, "_fetch_bars", raise_fetch)
    monkeypatch.setattr(data_fetcher.fh_fetcher, "fetch", raise_finnhub)
    monkeypatch.setattr(data_fetcher, "fetch_minute_yfinance", fake_yf)
    monkeypatch.setattr(data_fetcher, "is_market_open", lambda: True)

    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = start + pd.Timedelta(minutes=1)
    df = data_fetcher.get_minute_df("AAPL", start, end)
    assert called == ["AAPL"]
    assert not df.empty


def test_empty_bars_handled(monkeypatch):
    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = start + pd.Timedelta(minutes=1)

    monkeypatch.setattr(data_fetcher, "_fetch_bars", lambda *a, **k: pd.DataFrame())
    fallback = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]},
        index=[start],
    )
    monkeypatch.setattr(data_fetcher.fh_fetcher, "fetch", lambda *a, **k: fallback)
    monkeypatch.setattr(data_fetcher, "is_market_open", lambda: True)

    df = data_fetcher.get_minute_df("AAPL", start, end)
    pd.testing.assert_frame_equal(df, fallback)


def test_fetch_bars_empty_uses_last_bar(monkeypatch):
    class Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"bars": []}

    last = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]},
        index=[pd.Timestamp("2023-01-01", tz="UTC")],
    )

    monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(data_fetcher, "get_last_available_bar", lambda s: last)

    df = data_fetcher._fetch_bars(
        "AAPL",
        pd.Timestamp("2023-01-02", tz="UTC"),
        pd.Timestamp("2023-01-02", tz="UTC"),
        "1Day",
        "iex",
    )

    assert not df.empty and df.equals(last)

# AI-AGENT-REF: Replaced unsafe exec() with direct import from core module
from ai_trading.core.bot_engine import fetch_minute_df_safe


def test_fetch_minute_df_safe_no_retry(monkeypatch):
    calls = []

    def fake_get(sym, start, end):
        calls.append(1)
        return pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2023-01-01")])

    monkeypatch.setattr("ai_trading.core.bot_engine.get_minute_df", fake_get)
    result = fetch_minute_df_safe("AAPL")
    assert len(calls) == 1
    assert not result.empty


def test_fetch_minute_df_safe_raises(monkeypatch, caplog):
    from ai_trading.data_fetcher import DataFetchError
    monkeypatch.setattr("ai_trading.core.bot_engine.get_minute_df", lambda *a, **k: pd.DataFrame())
    caplog.set_level("ERROR")
    with pytest.raises(DataFetchError):
        fetch_minute_df_safe("AAPL")
    assert any("empty DataFrame" in r.message for r in caplog.records)
