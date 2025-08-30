import sys
import types

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("requests")
from ai_trading.utils.device import TORCH_AVAILABLE
if not TORCH_AVAILABLE:
    pytest.skip("torch not installed", allow_module_level=True)
# Minimal stubs so importing bot_engine succeeds without optional deps
mods = [
    "sklearn",
    "pandas_ta",
    "pandas_market_calendars",
    "schedule",
    "portalocker",
    "alpaca",
    "alpaca.trading.client",
    "alpaca.data.timeframe",
    "alpaca.data.requests",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "finnhub",
    "pybreaker",
    "ratelimit",
    "ai_trading.execution",
    "ai_trading.capital_scaling",
    # "strategy_allocator",  # AI-AGENT-REF: Don't mock this, it interferes with other tests
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

sys.modules["alpaca"].TradingClient = object
sys.modules["alpaca"].APIError = Exception
sys.modules.setdefault("alpaca.trading", types.ModuleType("alpaca.trading"))
sys.modules["alpaca.trading.client"] = types.ModuleType("alpaca.trading.client")
sys.modules["alpaca.trading.client"].TradingClient = object
sys.modules["alpaca.trading.client"].APIError = Exception
sys.modules.setdefault("alpaca.data", types.ModuleType("alpaca.data"))
sys.modules.setdefault("alpaca.data.timeframe", types.ModuleType("alpaca.data.timeframe"))
sys.modules.setdefault("alpaca.data.requests", types.ModuleType("alpaca.data.requests"))
sys.modules["alpaca.data.timeframe"].TimeFrame = object
sys.modules["alpaca.data.timeframe"].TimeFrameUnit = object
sys.modules["alpaca.data.requests"].StockBarsRequest = object
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = object
sys.modules["alpaca.data"].TimeFrame = sys.modules["alpaca.data.timeframe"].TimeFrame
sys.modules["alpaca.data"].StockBarsRequest = sys.modules["alpaca.data.requests"].StockBarsRequest

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
sys.modules["prometheus_client"].start_http_server = lambda *a, **k: None
sys.modules["prometheus_client"].Counter = lambda *a, **k: None
sys.modules["prometheus_client"].Gauge = lambda *a, **k: None
sys.modules["prometheus_client"].Histogram = lambda *a, **k: None
sys.modules["metrics_logger"].log_metrics = lambda *a, **k: None
sys.modules["finnhub"].FinnhubAPIException = Exception
sys.modules["finnhub"].Client = lambda *a, **k: None
# AI-AGENT-REF: Don't mock strategy_allocator to avoid test interference
# sys.modules["strategy_allocator"].StrategyAllocator = object
sys.modules.setdefault("ratelimit", types.ModuleType("ratelimit"))
sys.modules["ratelimit"].limits = lambda *a, **k: lambda f: f
sys.modules["ratelimit"].sleep_and_retry = lambda f: f

class _DummyBreaker:
    def __init__(self, *a, **k):
        pass

    def __call__(self, func):
        return func

sys.modules["pybreaker"].CircuitBreaker = _DummyBreaker

# AI-AGENT-REF: Remove ai_trading.main import that causes deep torch dependency chain
# from ai_trading.main import main  # Not used in this test, causes torch import issues
from ai_trading.core.bot_engine import pre_trade_health_check


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
