import sys
import types
from pathlib import Path

import pandas as pd

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
    "ai_trading.capital_scaling",
    "strategy_allocator",
]
for name in mods:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# Provide a minimal CapitalScalingEngine so bot imports succeed and
# tests can call ``update`` without errors.
if "ai_trading.capital_scaling" in sys.modules:
    class _CapScaler:
        def __init__(self, *a, **k):
            pass

        def update(self, *a, **k):
            pass

    sys.modules["ai_trading.capital_scaling"].CapitalScalingEngine = _CapScaler

sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
if "pandas_market_calendars" in sys.modules:
    sys.modules["pandas_market_calendars"].get_calendar = (
        lambda *a, **k: types.SimpleNamespace(
            schedule=lambda *a, **k: pd.DataFrame()
        )
    )
if "pandas_ta" in sys.modules:
    sys.modules["pandas_ta"].atr = lambda *a, **k: pd.Series([0])
    sys.modules["pandas_ta"].rsi = lambda *a, **k: pd.Series([0])

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
class _Flask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def decorator(func):
            return func

        return decorator

    def run(self, *a, **k):
        pass

sys.modules["flask"].Flask = _Flask
exc_mod = types.ModuleType("requests.exceptions")
exc_mod.HTTPError = Exception
exc_mod.RequestException = Exception
sys.modules["requests"].exceptions = exc_mod
sys.modules["requests"].get = lambda *a, **k: None
sys.modules["requests.exceptions"] = exc_mod
sys.modules["requests"].RequestException = Exception
sys.modules["alpaca_trade_api"].REST = object
sys.modules["alpaca_trade_api"].APIError = Exception
sys.modules["alpaca.common.exceptions"].APIError = Exception
class _RF:
    def __init__(self, *a, **k):
        pass

sys.modules["sklearn.ensemble"].RandomForestClassifier = _RF
class _Ridge:
    def __init__(self, *a, **k):
        pass

class _BR:
    def __init__(self, *a, **k):
        pass

sys.modules["sklearn.linear_model"].Ridge = _Ridge
sys.modules["sklearn.linear_model"].BayesianRidge = _BR
class _PCA:
    def __init__(self, *a, **k):
        pass

sys.modules["sklearn.decomposition"].PCA = _PCA
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

    def get_stock_bars(self, *a, **k):
        return types.SimpleNamespace(
            df=pd.DataFrame({"high": [1], "low": [1], "close": [1]})
        )

sys.modules["alpaca.data.historical"].StockHistoricalDataClient = _DummyDataClient
sys.modules["alpaca.data.models"].Quote = object
class _DummyReq:
    def __init__(self, *a, **k):
        pass

sys.modules["alpaca.data.requests"].StockBarsRequest = _DummyReq
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = _DummyReq
class _DummyTimeFrame:
    Day = object()

sys.modules["alpaca.data.timeframe"].TimeFrame = _DummyTimeFrame
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

import bot_engine as bot


def test_compute_time_range():
    """compute_time_range should span the requested minutes."""
    start, end = bot.compute_time_range(5)
    assert (end - start).total_seconds() == 300


def test_get_latest_close_edge_cases():
    """get_latest_close handles missing data gracefully."""
    assert bot.get_latest_close(pd.DataFrame()) == 0.0
    assert bot.get_latest_close(None) == 0.0
    df = pd.DataFrame({"close": [1.5]}, index=[pd.Timestamp("2024-01-01")])
    assert bot.get_latest_close(df) == 1.5


def test_fetch_minute_df_safe_market_closed(monkeypatch):
    """No data is returned when the market is closed."""
    monkeypatch.setattr(bot, "market_is_open", lambda now=None: False)
    result = bot.fetch_minute_df_safe("AAPL")
    assert result.empty


def test_fetch_minute_df_safe_open(monkeypatch):
    """DataFrame is returned when the market is open."""
    monkeypatch.setattr(bot, "market_is_open", lambda now=None: True)
    df = pd.DataFrame({"close": [1]}, index=[pd.Timestamp("2024-01-01")])
    monkeypatch.setattr(bot, "get_minute_df", lambda symbol, start_date, end_date: df)
    result = bot.fetch_minute_df_safe("AAPL")
    pd.testing.assert_frame_equal(result, df)
