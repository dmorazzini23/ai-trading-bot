import sys
import types
from typing import Any, cast

import pytest

pd = pytest.importorskip("pandas")
sys = cast(Any, sys)


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return cast(types.ModuleType, module)


def _set_module_attr(module_name: str, attr_name: str, value: Any) -> None:
    setattr(_ensure_module(module_name), attr_name, value)
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
    "alpaca.data",
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
    "ai_trading.capital_scaling",
    "strategy_allocator",
]
_PATCHED_MODULES = set(mods) | {"requests.exceptions", "flask.app"}
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _PATCHED_MODULES}


@pytest.fixture(scope="module", autouse=True)
def _restore_module_state_after_tests():
    yield
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
            continue
        sys.modules[name] = original


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

        def __call__(self, size):
            return size

        def scale_position(self, size):
            return size

    _set_module_attr("ai_trading.capital_scaling", "CapitalScalingEngine", _CapScaler)

sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
if "pandas_market_calendars" in sys.modules:
    _set_module_attr("pandas_market_calendars", "get_calendar", (
        lambda *a, **k: types.SimpleNamespace(
            schedule=lambda *a, **k: pd.DataFrame()
        )
    ))
if "pandas_ta" in sys.modules:
    _set_module_attr("pandas_ta", "atr", lambda *a, **k: pd.Series([0]))
    _set_module_attr("pandas_ta", "rsi", lambda *a, **k: pd.Series([0]))

# Provide required attributes for some stubs
_set_module_attr("pipeline", "model_pipeline", lambda *a, **k: None)
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("urllib3", types.ModuleType("urllib3"))


class _HTTPWarning(Warning):
    pass


class _SystemTimeWarning(Warning):
    pass


_set_module_attr("urllib3", "exceptions", types.SimpleNamespace(
    HTTPError=Exception,
    HTTPWarning=_HTTPWarning,
    SystemTimeWarning=_SystemTimeWarning,
))
sys.modules.setdefault("bs4", types.ModuleType("bs4"))
_set_module_attr("bs4", "BeautifulSoup", lambda *a, **k: None)
flask_mod = sys.modules.setdefault("flask", types.ModuleType("flask"))
flask_app_mod = sys.modules.setdefault("flask.app", types.ModuleType("flask.app"))


def _jsonify(payload=None, *args, **kwargs):
    if payload is not None:
        return payload
    if args:
        return args[0] if len(args) == 1 else list(args)
    return kwargs


class _Flask:
    def __init__(self, *a, **k):
        self._routes = {}
        self.config = {}

    def route(self, path, *a, **k):
        def decorator(func):
            self._routes[path] = func
            return func

        return decorator

    def run(self, *a, **k):
        pass

    def test_client(self):
        app = self

        class _Response:
            def __init__(self, data, status=200):
                self._data = data
                self.status_code = status

            def get_json(self):
                return self._data

        class _Client:
            def get(self, path):
                handler = app._routes[path]
                result = handler()
                status = 200
                data = result
                if isinstance(result, tuple):
                    data = result[0]
                    if len(result) > 1:
                        status = result[1]
                return _Response(data, status)

        return _Client()


if not hasattr(flask_mod, "Flask"):
    setattr(flask_mod, "Flask", _Flask)
if not hasattr(flask_app_mod, "Flask"):
    setattr(flask_app_mod, "Flask", _Flask)
if not hasattr(flask_mod, "jsonify"):
    setattr(flask_mod, "jsonify", _jsonify)
exc_mod = types.ModuleType("requests.exceptions")
setattr(exc_mod, "HTTPError", Exception)
setattr(exc_mod, "RequestException", Exception)
_set_module_attr("requests", "exceptions", exc_mod)
_set_module_attr("requests", "get", lambda *a, **k: None)
sys.modules["requests.exceptions"] = exc_mod
_set_module_attr("requests", "RequestException", Exception)
_set_module_attr("alpaca", "TradingClient", object)
_set_module_attr("alpaca", "APIError", Exception)
class _RF:
    def __init__(self, *a, **k):
        pass

_set_module_attr("sklearn.ensemble", "RandomForestClassifier", _RF)
class _Ridge:
    def __init__(self, *a, **k):
        pass

class _BR:
    def __init__(self, *a, **k):
        pass

_set_module_attr("sklearn.linear_model", "Ridge", _Ridge)
_set_module_attr("sklearn.linear_model", "BayesianRidge", _BR)
class _PCA:
    def __init__(self, *a, **k):
        pass

_set_module_attr("sklearn.decomposition", "PCA", _PCA)
_set_module_attr("alpaca.trading.client", "TradingClient", object)
_set_module_attr("alpaca.trading.client", "APIError", Exception)
class _DummyReq:
    def __init__(self, *a, **k):
        pass

_set_module_attr("alpaca.data.requests", "StockBarsRequest", _DummyReq)
_set_module_attr("alpaca.data.requests", "StockLatestQuoteRequest", _DummyReq)
class _DummyTimeFrame:
    Day = object()

_set_module_attr("alpaca.data.timeframe", "TimeFrame", _DummyTimeFrame)
_set_module_attr("alpaca.data.timeframe", "TimeFrameUnit", object)
_set_module_attr("alpaca.data", "TimeFrame", _DummyTimeFrame)
_set_module_attr("alpaca.data", "StockBarsRequest", _DummyReq)
sys.modules["bs4"] = types.ModuleType("bs4")
_set_module_attr("bs4", "BeautifulSoup", lambda *a, **k: None)
_set_module_attr("prometheus_client", "start_http_server", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Counter", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Gauge", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Histogram", lambda *a, **k: None)
_set_module_attr("metrics_logger", "log_metrics", lambda *a, **k: None)
_set_module_attr("finnhub", "FinnhubAPIException", Exception)
_set_module_attr("finnhub", "Client", lambda *a, **k: None)
_set_module_attr("strategy_allocator", "StrategyAllocator", object)
sys.modules.setdefault("ratelimit", types.ModuleType("ratelimit"))
_set_module_attr("ratelimit", "limits", lambda *a, **k: lambda f: f)
_set_module_attr("ratelimit", "sleep_and_retry", lambda f: f)


class _DummyBreaker:
    def __init__(self, *a, **k):
        pass

    def __call__(self, func):
        return func

_set_module_attr("pybreaker", "CircuitBreaker", _DummyBreaker)

from ai_trading.core import bot_engine as bot


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
    """Error is raised when no data is returned."""
    monkeypatch.setattr(bot, "get_minute_df", lambda *a, **k: pd.DataFrame())
    with pytest.raises(bot.DataFetchError):
        bot.fetch_minute_df_safe("AAPL")


def test_fetch_minute_df_safe_open(monkeypatch):
    """DataFrame is returned when the market is open."""
    df = pd.DataFrame({"close": [1]}, index=[pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1)])
    monkeypatch.setattr(bot, "get_minute_df", lambda symbol, start_date, end_date: df)
    monkeypatch.setattr(bot, "_count_trading_minutes", lambda *a, **k: 1)
    monkeypatch.setattr(bot, "_expected_minute_bars_window", lambda *a, **k: 1)
    monkeypatch.setattr(bot.staleness, "_ensure_data_fresh", lambda *a, **k: None)
    result = bot.fetch_minute_df_safe("AAPL")
    pd.testing.assert_frame_equal(result, df)
