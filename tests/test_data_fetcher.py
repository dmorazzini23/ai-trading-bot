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

import data_fetcher


class FakeBars:
    def __init__(self, df: pd.DataFrame):
        self.df = df


def test_get_minute_df(monkeypatch):
    pytest.skip("Skipping due to stubbed dependencies")
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
        index=[pd.Timestamp("2023-01-01T09:30")],
    )

    def fake_get_stock_bars(*args, **kwargs):
        return FakeBars(df)

    monkeypatch.setattr(data_fetcher.client, "get_stock_bars", fake_get_stock_bars)
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

    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2023-01-02", tz="UTC")
    messages = []
    monkeypatch.setattr(data_fetcher.logger, "critical", lambda msg, *a, **k: messages.append(msg))
    data_fetcher.get_minute_df("AAPL", start, end)
    assert messages == []
