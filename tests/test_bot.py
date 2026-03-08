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
    "strategy_allocator",
]
_PATCHED_MODULES = set(mods) | {"requests.exceptions"}
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
        setattr(sys.modules[name], "__spec__", types.SimpleNamespace())
_set_module_attr("pipeline", "model_pipeline", lambda *a, **k: None)

class _Flask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def decorator(func):
            return func

        return decorator

    def run(self, *a, **k):
        pass

_set_module_attr("flask", "Flask", _Flask)
_set_module_attr("flask", "jsonify", lambda *a, **k: None)
_set_module_attr("flask", "Response", object)
_set_module_attr("requests", "get", lambda *a, **k: None)
exc_mod = types.ModuleType("requests.exceptions")
setattr(exc_mod, "HTTPError", Exception)
setattr(exc_mod, "RequestException", Exception)
_set_module_attr("requests", "exceptions", exc_mod)
sys.modules["requests.exceptions"] = exc_mod
_set_module_attr("requests", "RequestException", Exception)
sys.modules["urllib3"] = types.ModuleType("urllib3")
_set_module_attr("urllib3", "exceptions", types.SimpleNamespace(HTTPError=Exception))
_set_module_attr("alpaca", "TradingClient", object)
_set_module_attr("alpaca", "APIError", Exception)
_set_module_attr("sklearn.ensemble", "RandomForestClassifier", object)
_set_module_attr("sklearn.ensemble", "GradientBoostingClassifier", object)
_set_module_attr("sklearn.linear_model", "Ridge", object)
_set_module_attr("sklearn.linear_model", "BayesianRidge", object)
_set_module_attr("sklearn.decomposition", "PCA", object)
_set_module_attr("sklearn.metrics", "accuracy_score", lambda *a, **k: 0)
_set_module_attr("sklearn.model_selection", "train_test_split", lambda *a, **k: ([], []))
_set_module_attr("sklearn.preprocessing", "StandardScaler", object)
_set_module_attr("prometheus_client", "REGISTRY", object())
_set_module_attr("prometheus_client", "CollectorRegistry", object)
_set_module_attr("prometheus_client", "Summary", object)


class _FakeREST:
    def __init__(self, *a, **k):
        pass


_set_module_attr("alpaca.trading.client", "TradingClient", _FakeREST)
_set_module_attr("alpaca.trading.client", "APIError", Exception)
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

_set_module_attr("sklearn.ensemble", "RandomForestClassifier", _RF)
_set_module_attr("sklearn.linear_model", "Ridge", _Ridge)
_set_module_attr("sklearn.linear_model", "BayesianRidge", _BR)
_set_module_attr("sklearn.decomposition", "PCA", _PCA)
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

_set_module_attr("bs4", "BeautifulSoup", lambda *a, **k: None)
_set_module_attr("prometheus_client", "start_http_server", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Counter", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Gauge", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Histogram", lambda *a, **k: None)
_set_module_attr("metrics_logger", "log_metrics", lambda *a, **k: None)
_set_module_attr("finnhub", "FinnhubAPIException", Exception)
_set_module_attr("finnhub", "Client", lambda *a, **k: None)
_set_module_attr("strategy_allocator", "StrategyAllocator", object)
class _DummyBreaker:
    def __init__(self, *a, **k):
        pass

    def __call__(self, func):
        return func

_set_module_attr("pybreaker", "CircuitBreaker", _DummyBreaker)
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
setattr(_cap_mod, "CapitalScalingEngine", _CapScaler)
if "pandas_market_calendars" in sys.modules:
    _set_module_attr("pandas_market_calendars", "get_calendar", (
        lambda *a, **k: types.SimpleNamespace(
            schedule=lambda *a, **k: pd.DataFrame()
        )
    ))
if "pandas_ta" in sys.modules:
    _set_module_attr("pandas_ta", "atr", lambda *a, **k: pd.Series([0]))
    _set_module_attr("pandas_ta", "rsi", lambda *a, **k: pd.Series([0]))
    _set_module_attr("pandas_ta", "obv", lambda *a, **k: pd.Series([0]))
    _set_module_attr("pandas_ta", "vwap", lambda *a, **k: pd.Series([0]))

from ai_trading.core import bot_engine as bot


def test_screen_candidates_empty(monkeypatch):
    """screen_candidates returns an empty list when none pass."""
    monkeypatch.setattr(bot, "screen_universe", lambda candidates, runtime: [])

    # Create a mock runtime object
    from unittest.mock import Mock
    mock_runtime = Mock()

    assert bot.screen_candidates(mock_runtime, ["AAA"]) == []


def test_screen_universe_atr_fallback(monkeypatch):
    """screen_universe falls back to internal ATR when pandas_ta is missing."""
    import ai_trading.indicators as indicators

    monkeypatch.setattr(bot, "ta", types.SimpleNamespace(_failed=True))
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {})
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


def test_screen_universe_refetches_for_missing_atr(monkeypatch):
    """screen_universe fetches more history when ATR cannot be computed."""
    import ai_trading.data.fetch as data_fetch

    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {})
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "is_valid_ohlcv", lambda df: True)
    monkeypatch.setattr(bot, "_validate_market_data_quality", lambda df, s: {"valid": True})

    rows_short = bot.ATR_LENGTH - 1
    df_short = pd.DataFrame(
        {
            "high": range(rows_short),
            "low": range(rows_short),
            "close": range(rows_short),
            "volume": [200_000] * rows_short,
        }
    )
    rows_long = bot.ATR_LENGTH + 5
    df_long = pd.DataFrame(
        {
            "high": range(rows_long),
            "low": range(rows_long),
            "close": range(rows_long),
            "volume": [200_000] * rows_long,
        }
    )

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            return df_short

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    called = {"extended": False}

    def fake_get_daily_df(symbol, start, end):
        called["extended"] = True
        return df_long

    monkeypatch.setattr(data_fetch, "get_daily_df", fake_get_daily_df)
    def fake_atr(h, l, c, length=bot.ATR_LENGTH):
        if len(h) < length:
            return pd.Series([])
        return pd.Series([1.0] * len(h))

    monkeypatch.setattr(bot, "ta", types.SimpleNamespace(atr=fake_atr))

    result = bot.screen_universe(["AAA"], runtime)

    assert result == ["AAA"]
    assert called["extended"] is True


def test_screen_universe_skips_when_atr_still_missing(monkeypatch):
    """screen_universe skips symbols when ATR remains unavailable."""
    import ai_trading.data.fetch as data_fetch

    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {})
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)
    monkeypatch.setattr(bot, "is_valid_ohlcv", lambda df: True)
    monkeypatch.setattr(bot, "_validate_market_data_quality", lambda df, s: {"valid": True})

    rows_short = bot.ATR_LENGTH - 1
    df_short = pd.DataFrame(
        {
            "high": range(rows_short),
            "low": range(rows_short),
            "close": range(rows_short),
            "volume": [200_000] * rows_short,
        }
    )

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            return df_short

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    called = {"extended": False}

    def fake_get_daily_df(symbol, start, end):
        called["extended"] = True
        return df_short

    monkeypatch.setattr(data_fetch, "get_daily_df", fake_get_daily_df)
    monkeypatch.setattr(
        bot,
        "ta",
        types.SimpleNamespace(
            atr=lambda h, l, c, length=bot.ATR_LENGTH: pd.Series([])
        ),
    )

    result = bot.screen_universe(["AAA"], runtime)

    assert result == []
    assert called["extended"] is True


def test_screen_universe_reuses_cached_candidates_when_refetch_window_open(monkeypatch):
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {"AAA": 1.25})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {"AAA": bot.time.time()})
    monkeypatch.setattr(bot, "_SCREEN_ROTATE_UNSEEN_ENABLED", False)
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(bot, "is_market_open", lambda: True)

    rows = bot.ATR_LENGTH + 3
    df_spy = pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [500_000] * rows,
        }
    )

    class DummyFetcher:
        def __init__(self):
            self.requested: list[str] = []

        def get_daily_df(self, runtime, sym):
            self.requested.append(sym)
            if sym == "SPY":
                return df_spy
            raise AssertionError(f"unexpected fetch for {sym}")

    fetcher = DummyFetcher()
    runtime = types.SimpleNamespace(data_fetcher=fetcher)

    selected = bot.screen_universe(["AAA"], runtime)

    assert selected == ["AAA"]
    assert fetcher.requested == ["SPY"]


def test_screen_universe_does_not_rotate_unseen_when_window_throttled(monkeypatch):
    monkeypatch.setattr(bot, "_SCREEN_CACHE", {})
    monkeypatch.setattr(bot, "_LAST_SCREEN_FETCH", {"AAA": bot.time.time()})
    monkeypatch.setattr(bot, "_SCREEN_ROTATE_UNSEEN_ENABLED", False)
    monkeypatch.setattr(bot.time, "sleep", lambda *a, **k: None)

    class DummyFetcher:
        def get_daily_df(self, runtime, sym):
            raise AssertionError(f"unexpected fetch for {sym}")

    runtime = types.SimpleNamespace(data_fetcher=DummyFetcher())

    selected = bot.screen_universe(["AAA"], runtime)

    assert selected == []


def test_resolve_prepare_symbol_limit_prefers_explicit_setting(monkeypatch):
    monkeypatch.setenv("AI_TRADING_PREPARE_SYMBOL_LIMIT", "12")
    monkeypatch.setenv("MAX_SYMBOLS_PER_CYCLE", "3")
    assert bot._resolve_prepare_symbol_limit() == 12


def test_resolve_prepare_symbol_limit_falls_back_to_max_symbols(monkeypatch):
    monkeypatch.setenv("AI_TRADING_PREPARE_SYMBOL_LIMIT", "0")
    monkeypatch.setenv("MAX_SYMBOLS_PER_CYCLE", "4")
    assert bot._resolve_prepare_symbol_limit() == 4
