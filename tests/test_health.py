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
import ai_trading.data.fetch as data_fetch
import ai_trading.alpaca_api as alpaca_api


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
    summary = pre_trade_health_check(ctx, ["AAA"], min_rows=30)
    assert summary["failures"] == ["AAA"]


def test_health_check_succeeds(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "30")
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020", periods=30, tz="UTC"),
            "open": [1] * 30,
            "high": [1] * 30,
            "low": [1] * 30,
            "close": [1] * 30,
            "volume": [1] * 30,
        }
    )
    ctx = DummyCtx(df)
    summary = pre_trade_health_check(ctx, ["AAA"], min_rows=30)
    assert summary["missing_columns"] == []
    assert summary["failures"] == []
    assert summary["checked"] == 1


def test_health_check_accepts_datetime_index(monkeypatch):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "5")
    df = pd.DataFrame(
        {
            "open": [1] * 5,
            "high": [1] * 5,
            "low": [1] * 5,
            "close": [1] * 5,
            "volume": [1] * 5,
        },
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
    )
    ctx = DummyCtx(df)
    summary = pre_trade_health_check(ctx, ["AAA"], min_rows=5)
    assert summary["missing_columns"] == []
    assert summary["failures"] == []
    assert summary["checked"] == 1


def test_get_daily_df_normalizes_columns(monkeypatch):
    df = pd.DataFrame({"t": [0], "o": [1], "h": [1], "l": [1], "c": [1], "v": [1]})
    monkeypatch.setattr(data_fetch, "should_import_alpaca_sdk", lambda: True)
    monkeypatch.setattr(alpaca_api, "get_bars_df", lambda *a, **k: df)
    out = data_fetch.get_daily_df("AAA")
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in out.columns


def test_get_daily_df_uses_backup_when_primary_normalization_empties(monkeypatch):
    primary_df = pd.DataFrame(
        {
            "timestamp": [None],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100],
        }
    )
    fallback_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "open": [1.0, 1.2],
            "high": [1.1, 1.25],
            "low": [0.95, 1.15],
            "close": [1.05, 1.22],
            "volume": [100, 150],
        }
    )

    monkeypatch.setattr(data_fetch, "should_import_alpaca_sdk", lambda: True)
    monkeypatch.setattr(alpaca_api, "get_bars_df", lambda *a, **k: primary_df)
    monkeypatch.setattr(
        data_fetch,
        "_backup_get_bars",
        lambda *a, **k: fallback_df,
    )

    out = data_fetch.get_daily_df("AAA")

    assert len(out) == len(fallback_df)
    pd.testing.assert_index_equal(out.index, fallback_df.set_index("timestamp").index)


def test_ensure_schema_then_normalize_restores_timestamp_column():
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.2, 1.25],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.2],
            "volume": [100, 120, 110],
        },
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )

    ensured = data_fetch.ensure_ohlcv_schema(df, source="test", frequency="1Day")
    normalized = data_fetch.normalize_ohlcv_df(ensured, include_columns=("timestamp",))

    assert "timestamp" in normalized.columns
    expected = pd.Series(normalized.index, index=normalized.index, name="timestamp")
    pd.testing.assert_series_equal(normalized["timestamp"], expected)
