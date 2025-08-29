import sys
import types

import pytest

pd = pytest.importorskip("pandas")

dummy = types.ModuleType("dummy")
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
    "alpaca.data",
    "alpaca.data.timeframe",
    "alpaca.data.requests",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.preprocessing",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "finnhub",
    "pybreaker",
    "ai_trading.execution",
    "strategy_allocator",
]
for name in mods:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__spec__ = types.SimpleNamespace()
sys.modules["pipeline"].model_pipeline = lambda *a, **k: None

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
sys.modules["requests"].get = lambda *a, **k: None
exc_mod = types.ModuleType("requests.exceptions")
exc_mod.HTTPError = Exception
exc_mod.RequestException = Exception
sys.modules["requests"].exceptions = exc_mod
sys.modules["requests.exceptions"] = exc_mod
sys.modules["requests"].RequestException = Exception
sys.modules["urllib3"] = types.ModuleType("urllib3")
sys.modules["urllib3"].exceptions = types.SimpleNamespace(HTTPError=Exception)
sys.modules["alpaca"].TradingClient = object
sys.modules["alpaca"].APIError = Exception
sys.modules["sklearn.ensemble"].RandomForestClassifier = object
sys.modules["sklearn.ensemble"].GradientBoostingClassifier = object
sys.modules["sklearn.linear_model"].Ridge = object
sys.modules["sklearn.linear_model"].BayesianRidge = object
sys.modules["sklearn.decomposition"].PCA = object
sys.modules["sklearn.metrics"].accuracy_score = lambda *a, **k: 0
sys.modules["sklearn.model_selection"].train_test_split = lambda *a, **k: ([], [])
sys.modules["sklearn.preprocessing"].StandardScaler = object
sys.modules["prometheus_client"].REGISTRY = object()
sys.modules["prometheus_client"].CollectorRegistry = object
sys.modules["prometheus_client"].Summary = object


class _FakeREST:
    def __init__(self, *a, **k):
        pass


sys.modules["alpaca.trading.client"].TradingClient = _FakeREST
sys.modules["alpaca.trading.client"].APIError = Exception
class _DummyTradingClient:
    def __init__(self, *a, **k):
        pass
class _RF:
    def __init__(self, *a, **k):
        pass

class _Ridge:
    def __init__(self, *a, **k):
        pass

class _BR:
    def __init__(self, *a, **k):
        pass

class _PCA:
    def __init__(self, *a, **k):
        pass

sys.modules["sklearn.ensemble"].RandomForestClassifier = _RF
sys.modules["sklearn.linear_model"].Ridge = _Ridge
sys.modules["sklearn.linear_model"].BayesianRidge = _BR
sys.modules["sklearn.decomposition"].PCA = _PCA
class _DummyReq:
    def __init__(self, *a, **k):
        pass

sys.modules["alpaca.data.requests"].StockBarsRequest = _DummyReq
sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = _DummyReq
class _DummyTimeFrame:
    Day = object()

sys.modules["alpaca.data.timeframe"].TimeFrame = _DummyTimeFrame
sys.modules["alpaca.data.timeframe"].TimeFrameUnit = object
sys.modules["alpaca.data"].TimeFrame = _DummyTimeFrame
sys.modules["alpaca.data"].StockBarsRequest = _DummyReq

sys.modules["bs4"].BeautifulSoup = lambda *a, **k: None
sys.modules["prometheus_client"].start_http_server = lambda *a, **k: None
sys.modules["prometheus_client"].Counter = lambda *a, **k: None
sys.modules["prometheus_client"].Gauge = lambda *a, **k: None
sys.modules["prometheus_client"].Histogram = lambda *a, **k: None
sys.modules["metrics_logger"].log_metrics = lambda *a, **k: None
sys.modules["finnhub"].FinnhubAPIException = Exception
sys.modules["finnhub"].Client = lambda *a, **k: None
sys.modules["strategy_allocator"].StrategyAllocator = object
class _DummyBreaker:
    def __init__(self, *a, **k):
        pass

    def __call__(self, func):
        return func

sys.modules["pybreaker"].CircuitBreaker = _DummyBreaker
class _CapScaler:
    def __init__(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def __call__(self, size):
        return size

    def scale_position(self, size):
        return size
import ai_trading.capital_scaling as _cap_mod
_cap_mod.CapitalScalingEngine = _CapScaler
if "pandas_market_calendars" in sys.modules:
    sys.modules["pandas_market_calendars"].get_calendar = (
        lambda *a, **k: types.SimpleNamespace(
            schedule=lambda *a, **k: pd.DataFrame()
        )
    )
if "pandas_ta" in sys.modules:
    sys.modules["pandas_ta"].atr = lambda *a, **k: pd.Series([0])
    sys.modules["pandas_ta"].rsi = lambda *a, **k: pd.Series([0])
    sys.modules["pandas_ta"].obv = lambda *a, **k: pd.Series([0])
    sys.modules["pandas_ta"].vwap = lambda *a, **k: pd.Series([0])

from ai_trading.core import bot_engine as bot


def test_screen_candidates_empty(monkeypatch):
    """screen_candidates returns an empty list when none pass."""
    monkeypatch.setattr(bot, "load_tickers", lambda path=bot.TICKERS_FILE: ["AAA"])
    monkeypatch.setattr(bot, "screen_universe", lambda candidates, runtime: [])

    # Create a mock runtime object
    from unittest.mock import Mock
    mock_runtime = Mock()

    assert bot.screen_candidates(mock_runtime) == []


def test_screen_universe_atr_fallback(monkeypatch):
    """screen_universe falls back to internal ATR when pandas_ta is missing."""
    import ai_trading.indicators as indicators

    monkeypatch.setattr(bot, "ta", types.SimpleNamespace(_failed=True))
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "is_valid_ohlcv", lambda df: True)
    monkeypatch.setattr(bot, "_validate_market_data_quality", lambda df, s: {"valid": True})

    called = {}

    def fake_atr(h, l, c, period=14):
        called["used"] = True
        return pd.Series([1.0] * len(h))

    monkeypatch.setattr(indicators, "atr", fake_atr)

    rows = bot.ATR_LENGTH + 1
    df = pd.DataFrame(
        {
            "high": range(2, 2 + rows),
            "low": range(1, 1 + rows),
            "close": [x + 0.5 for x in range(1, 1 + rows)],
            "volume": [200_000] * rows,
        }
    )

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            return df

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    result = bot.screen_universe(["AAA"], runtime)

    assert result == ["AAA"]
    assert called.get("used") is True
