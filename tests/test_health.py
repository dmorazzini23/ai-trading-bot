import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Minimal stubs so importing bot_engine succeeds without optional deps
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
mods = [
    "sklearn",
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
    "pybreaker",
    "ratelimit",
    "trade_execution",
    "ai_trading.capital_scaling",
    "strategy_allocator",
    "torch",
]
for name in mods:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

if "sklearn" in sys.modules:
    sys.modules["sklearn"].ensemble = sys.modules["sklearn.ensemble"]
    sys.modules["sklearn"].linear_model = sys.modules["sklearn.linear_model"]
    sys.modules["sklearn"].decomposition = sys.modules["sklearn.decomposition"]

if "ai_trading.capital_scaling" in sys.modules:
    class _CapScaler:
        def __init__(self, *a, **k):
            pass

        def update(self, *a, **k):
            pass

        def __call__(self, size):
            return size

        def scale_position(self, size):
            return size

    sys.modules["ai_trading.capital_scaling"].CapitalScalingEngine = _CapScaler

sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
if "pandas_market_calendars" in sys.modules:
    sys.modules["pandas_market_calendars"].get_calendar = (
        lambda *a, **k: types.SimpleNamespace(schedule=lambda *a, **k: pd.DataFrame())
    )
if "pandas_ta" in sys.modules:
    sys.modules["pandas_ta"].atr = lambda *a, **k: pd.Series([0])
    sys.modules["pandas_ta"].rsi = lambda *a, **k: pd.Series([0])

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
sys.modules["flask"].jsonify = lambda *a, **k: None
sys.modules["flask"].Response = object
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
        return types.SimpleNamespace(df=pd.DataFrame({"high": [1], "low": [1], "close": [1]}))

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

# Minimal torch stub for bot_engine imports
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["torch"].manual_seed = lambda *a, **k: None
sys.modules["torch"].Tensor = object
sys.modules["torch"].tensor = lambda *a, **k: types.SimpleNamespace(detach=lambda: types.SimpleNamespace(numpy=lambda: [0.0]))
torch_nn = types.ModuleType("torch.nn")
torch_nn.Module = object
torch_nn.Sequential = lambda *a, **k: None
torch_nn.Linear = lambda *a, **k: None
torch_nn.ReLU = lambda *a, **k: None
torch_nn.Softmax = lambda *a, **k: None
sys.modules["torch.nn"] = torch_nn
torch_optim = types.ModuleType("torch.optim")
torch_optim.Adam = lambda *a, **k: None
sys.modules["torch.optim"] = torch_optim

from ai_trading.main import main
from bot_engine import pre_trade_health_check


class DummyFetcher:
    def __init__(self, df):
        self.df = df
    def get_daily_df(self, ctx, sym):
        return self.df

class DummyAPI:
    def get_account(self):
        return types.SimpleNamespace()

class DummyCtx:
    def __init__(self, df):
        self.data_fetcher = DummyFetcher(df)
        self.api = DummyAPI()


def test_health_check_empty_dataframe(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "30")
    ctx = DummyCtx(pd.DataFrame())
    summary = pre_trade_health_check(ctx, ["AAA"])
    assert summary["failures"] == ["AAA"]


def test_health_check_succeeds(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "30")
    df = pd.DataFrame({
        "open": [1] * 30,
        "high": [1] * 30,
        "low": [1] * 30,
        "close": [1] * 30,
        "volume": [1] * 30,
    })
    ctx = DummyCtx(df)
    pre_trade_health_check(ctx, ["AAA"])

