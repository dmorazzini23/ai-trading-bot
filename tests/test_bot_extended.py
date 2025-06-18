import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Ensure repository root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Minimal stubs so that importing bot succeeds without optional deps
mods = [
    "pandas_ta",
    "pandas_market_calendars",
    "requests",
    "urllib3",
    "bs4",
    "flask",
    "schedule",
    "portalocker",
    "alpaca",
    "alpaca.trading.client",
    "alpaca.trading.enums",
    "alpaca.trading.requests",
    "alpaca.trading.models",
    "alpaca_trade_api",
    "alpaca_trade_api.rest",
    "alpaca.data",
    "alpaca.trading.stream",
    "alpaca.data.historical",
    "alpaca.data.models",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
    "alpaca.common.exceptions",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "finnhub",
    "joblib",
    "pybreaker",
    "ratelimit",
    "trade_execution",
    "capital_scaling",
    "strategy_allocator",
]
for name in mods:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
sys.modules.setdefault("sentry_sdk", types.ModuleType("sentry_sdk"))
sys.modules["sentry_sdk"].init = lambda *a, **k: None

# Provide required attributes for some stubs
sys.modules["pipeline"].model_pipeline = lambda *a, **k: None
class _DummyStream:
    def __init__(self, *a, **k):
        pass
    def subscribe_trade_updates(self, *a, **k):
        pass

sys.modules["alpaca.trading.stream"].TradingStream = _DummyStream
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("urllib3", types.ModuleType("urllib3"))
sys.modules["urllib3"].exceptions = types.SimpleNamespace(HTTPError=Exception)
sys.modules.setdefault("bs4", types.ModuleType("bs4"))
sys.modules["bs4"].BeautifulSoup = lambda *a, **k: None
sys.modules.setdefault("flask", types.ModuleType("flask"))
sys.modules["flask"].Flask = object
exc_mod = types.ModuleType("requests.exceptions")
exc_mod.HTTPError = Exception
exc_mod.RequestException = Exception
sys.modules["requests"].exceptions = exc_mod
sys.modules["requests"].get = lambda *a, **k: None
sys.modules["requests.exceptions"] = exc_mod
sys.modules["alpaca_trade_api"].REST = object
sys.modules["alpaca_trade_api"].APIError = Exception
sys.modules["alpaca.common.exceptions"].APIError = Exception
sys.modules["sklearn.ensemble"].RandomForestClassifier = object
sys.modules["sklearn.linear_model"].Ridge = object
sys.modules["sklearn.linear_model"].BayesianRidge = object
sys.modules["sklearn.decomposition"].PCA = object
sys.modules["joblib"] = types.ModuleType("joblib")

sys.modules["alpaca_trade_api.rest"].REST = object
sys.modules["alpaca_trade_api.rest"].APIError = Exception
class _DummyTradingClient:
    def __init__(self, *a, **k):
        pass

sys.modules["alpaca.trading.client"].TradingClient = _DummyTradingClient
sys.modules["alpaca.trading.enums"].OrderSide = object
sys.modules["alpaca.trading.enums"].OrderStatus = object
sys.modules["alpaca.trading.enums"].QueryOrderStatus = object
sys.modules["alpaca.trading.enums"].TimeInForce = object
sys.modules["alpaca.trading.requests"].GetOrdersRequest = object
sys.modules["alpaca.trading.requests"].MarketOrderRequest = object
sys.modules["alpaca.trading.requests"].LimitOrderRequest = object
sys.modules["alpaca.trading.models"].Order = object
class _DummyDataClient:
    def __init__(self, *a, **k):
        pass

sys.modules["alpaca.data.historical"].StockHistoricalDataClient = _DummyDataClient
sys.modules["alpaca.data.models"].Quote = object
sys.modules["alpaca.data.requests"].StockBarsRequest = object
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = object
sys.modules["alpaca.data.timeframe"].TimeFrame = object
sys.modules["alpaca.data.timeframe"].TimeFrameUnit = object
sys.modules["bs4"] = types.ModuleType("bs4")
sys.modules["bs4"].BeautifulSoup = lambda *a, **k: None
sys.modules["prometheus_client"].start_http_server = lambda *a, **k: None
sys.modules["prometheus_client"].Counter = lambda *a, **k: None
sys.modules["prometheus_client"].Gauge = lambda *a, **k: None
sys.modules["prometheus_client"].Histogram = lambda *a, **k: None
sys.modules["metrics_logger"].log_metrics = lambda *a, **k: None
sys.modules["finnhub"].FinnhubAPIException = Exception
sys.modules["finnhub"].Client = lambda *a, **k: None
sys.modules["strategy_allocator"].StrategyAllocator = object
sys.modules.setdefault("ratelimit", types.ModuleType("ratelimit"))
sys.modules["ratelimit"].limits = lambda *a, **k: lambda f: f
sys.modules["ratelimit"].sleep_and_retry = lambda f: f
class _DummyBreaker:
    def __init__(self, *a, **k):
        pass
    def __call__(self, func):
        return func

sys.modules["pybreaker"].CircuitBreaker = _DummyBreaker

import bot


def test_compute_time_range():
    start, end = bot.compute_time_range(5)
    assert (end - start).total_seconds() == 300


def test_get_latest_close_edge_cases():
    assert bot.get_latest_close(pd.DataFrame()) == 0.0
    assert bot.get_latest_close(None) == 0.0
    df = pd.DataFrame({"close": [1.5]}, index=[pd.Timestamp("2024-01-01")])
    assert bot.get_latest_close(df) == 1.5


def test_fetch_minute_df_safe_market_closed(monkeypatch):
    monkeypatch.setattr(bot, "market_is_open", lambda now=None: False)
    result = bot.fetch_minute_df_safe("AAPL")
    assert result.empty


def test_fetch_minute_df_safe_open(monkeypatch):
    monkeypatch.setattr(bot, "market_is_open", lambda now=None: True)
    df = pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2024-01-01")])
    monkeypatch.setattr(bot, "get_minute_df", lambda symbol, start_date, end_date: df)
    result = bot.fetch_minute_df_safe("AAPL")
    pd.testing.assert_frame_equal(result, df)
