from tests.optdeps import require
require("pandas")
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    "alpaca_trade_api",
    "alpaca_trade_api.rest",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "finnhub",
    "pybreaker",
    "trade_execution",
    "ai_trading.capital_scaling",
    "strategy_allocator",
]
for name in mods:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
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
sys.modules["alpaca_trade_api"].REST = object
sys.modules["alpaca_trade_api"].APIError = Exception
sys.modules["sklearn.ensemble"].RandomForestClassifier = object
sys.modules["sklearn.linear_model"].Ridge = object
sys.modules["sklearn.linear_model"].BayesianRidge = object
sys.modules["sklearn.decomposition"].PCA = object


class _FakeREST:
    def __init__(self, *a, **k):
        pass


sys.modules["alpaca_trade_api.rest"].REST = _FakeREST
sys.modules["alpaca_trade_api.rest"].APIError = Exception
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

sys.modules["alpaca_trade_api.rest"].StockBarsRequest = _DummyReq
sys.modules["alpaca_trade_api.rest"].StockLatestQuoteRequest = _DummyReq
class _DummyTimeFrame:
    Day = object()

sys.modules["alpaca_trade_api.rest"].TimeFrame = _DummyTimeFrame
sys.modules["alpaca_trade_api.rest"].TimeFrameUnit = object

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

sys.modules["ai_trading.capital_scaling"].CapitalScalingEngine = _CapScaler
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
