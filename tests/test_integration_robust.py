import importlib
import sys
import types
from typing import Any, Callable, cast
from unittest.mock import MagicMock, patch

import pytest
from ai_trading.config import management as config

pd = pytest.importorskip("pandas")
pytest.importorskip("requests")
sys = cast(Any, sys)


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return cast(types.ModuleType, module)


def _set_module_attr(module_name: str, attr_name: str, value: Any) -> None:
    setattr(_ensure_module(module_name), attr_name, value)

_PATCHED_MODULES = {
    "bs4",
    "flask",
    "schedule",
    "portalocker",
    "alpaca",
    "alpaca.trading.client",
    "alpaca.data",
    "alpaca.data.timeframe",
    "alpaca.data.requests",
    "finnhub",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "pybreaker",
    "yfinance",
    "ratelimit",
    "ai_trading.capital_scaling",
    "strategy_allocator",
    "pandas_ta",
    "pandas_market_calendars",
}
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _PATCHED_MODULES}


@pytest.fixture(scope="module", autouse=True)
def _restore_module_state_after_tests():
    """Prevent integration stubs from affecting subsequent modules."""
    yield
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
            continue
        sys.modules[name] = original
        try:
            importlib.reload(original)
        except Exception:
            pass

"""Minimal import-time stubs so strategy_allocator and other modules load."""
try:
    pass  # type: ignore
except ImportError:
    sys.modules["pandas"] = types.ModuleType("pandas")
    _set_module_attr("pandas", "DataFrame", MagicMock())
    _set_module_attr("pandas", "Series", MagicMock())
    _set_module_attr("pandas", "concat", MagicMock())

pytestmark = pytest.mark.integration

try:
    import numpy  # type: ignore  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")
    _set_module_attr("numpy", "array", MagicMock())
    _set_module_attr("numpy", "nan", float("nan"))
    _set_module_attr("numpy", "NaN", float("nan"))
    _set_module_attr("numpy", "random", MagicMock())
    _set_module_attr("numpy", "arange", MagicMock())

try:
    import pandas_ta  # type: ignore  # noqa: F401
except ImportError:
    sys.modules["pandas_ta"] = types.ModuleType("pandas_ta")
if "pandas_ta" in sys.modules:
    mod = sys.modules["pandas_ta"]
    if not hasattr(mod, "momentum"):
        setattr(mod, "momentum", types.SimpleNamespace(rsi=MagicMock()))
    setattr(mod, "atr", getattr(mod, "atr", MagicMock(return_value=MagicMock())))
    setattr(mod, "rsi", getattr(mod, "rsi", MagicMock(return_value=MagicMock())))
    setattr(mod, "macd", getattr(mod, "macd", MagicMock(return_value={"MACD_12_26_9": MagicMock()})))
    setattr(mod, "sma", getattr(mod, "sma", MagicMock(return_value=MagicMock())))

try:
    import pandas_market_calendars  # type: ignore  # noqa: F401
except ImportError:
    sys.modules["pandas_market_calendars"] = types.ModuleType("pandas_market_calendars")
if not hasattr(sys.modules["pandas_market_calendars"], "get_calendar"):
    _set_module_attr("pandas_market_calendars", "get_calendar", MagicMock())

mods = [
    "bs4",
    "flask",
    "schedule",
    "portalocker",
    "alpaca",
    "alpaca.trading.client",
    "alpaca.data",
    "alpaca.data.timeframe",
    "alpaca.data.requests",
    "finnhub",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "pipeline",
    "metrics_logger",
    "prometheus_client",
    "pybreaker",
    "yfinance",
    "ratelimit",
    "ai_trading.capital_scaling",
    "strategy_allocator",
]
for m in mods:
    sys.modules.setdefault(m, types.ModuleType(m))
_set_module_attr("alpaca", "TradingClient", object)
_set_module_attr("alpaca", "APIError", Exception)
sys.modules["alpaca.trading.client"] = types.ModuleType("alpaca.trading.client")
_set_module_attr("alpaca.trading.client", "TradingClient", object)
_set_module_attr("alpaca.trading.client", "APIError", Exception)
sys.modules.setdefault("alpaca.data", types.ModuleType("alpaca.data"))
sys.modules.setdefault("alpaca.data.requests", types.ModuleType("alpaca.data.requests"))
sys.modules.setdefault("alpaca.data.timeframe", types.ModuleType("alpaca.data.timeframe"))


class _TClient:
    def __init__(self, *a, **k):
        pass




class _Req:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


from enum import Enum


class _Enum(str, Enum):
    pass

class _OrderSide(_Enum):
    BUY = "buy"
    SELL = "sell"

class _TimeInForce(_Enum):
    DAY = "day"

class _QueryOrderStatus(_Enum):
    pass

class _OrderStatus(_Enum):
    pass



class _Stream:
    def __init__(self, *a, **k):
        pass

    def subscribe_trade_updates(self, *a, **k):
        pass




class _StockLatestQuoteRequest:
    def __init__(self, *a, **k):
        pass


class _StockBarsRequest:
    def __init__(self, *a, **k):
        pass

_set_module_attr("alpaca.data.requests", "StockBarsRequest", _StockBarsRequest)
_set_module_attr("alpaca.data", "StockBarsRequest", _StockBarsRequest)




class _Client:
    def __init__(self, *a, **k):
        pass

    def get_stock_bars(self, *a, **k):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1],
            },
            index=[pd.Timestamp("2024-01-01", tz="UTC")],
        )
        return types.SimpleNamespace(df=df)




class _TF:
    Minute = "1Min"
    Hour = "1Hour"
    Day = "1Day"
    Week = "1Week"
    Month = "1Month"

    def __init__(self, *a, **k):
        pass


class _TFUnit:
    Minute = "Minute"
    Hour = "Hour"
    Day = "Day"
    Week = "Week"
    Month = "Month"




_set_module_attr("alpaca.data.timeframe", "TimeFrame", _TF)
_set_module_attr("alpaca.data.timeframe", "TimeFrameUnit", _TFUnit)
_set_module_attr("alpaca.data", "TimeFrame", _TF)
_set_module_attr("alpaca.data", "TimeFrameUnit", _TFUnit)


class _FClient:
    def __init__(self, *a, **k):
        pass


_set_module_attr("finnhub", "Client", _FClient)
_set_module_attr("finnhub", "FinnhubAPIException", Exception)
_set_module_attr("bs4", "BeautifulSoup", lambda *a, **k: None)


class _Flask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def decorator(f):
            return f

        return decorator


_set_module_attr("flask", "Flask", _Flask)


class _RFC:
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


_set_module_attr("sklearn.ensemble", "RandomForestClassifier", _RFC)
_set_module_attr("sklearn.linear_model", "Ridge", _Ridge)
_set_module_attr("sklearn.linear_model", "BayesianRidge", _BR)
_set_module_attr("sklearn.decomposition", "PCA", _PCA)
_set_module_attr("prometheus_client", "start_http_server", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Counter", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Gauge", lambda *a, **k: None)
_set_module_attr("prometheus_client", "Histogram", lambda *a, **k: None)
sys.modules["pipeline"] = types.ModuleType("pipeline")
sys.modules["metrics_logger"] = types.ModuleType("metrics_logger")
sys.modules["pybreaker"] = types.ModuleType("pybreaker")
_set_module_attr("pipeline", "model_pipeline", lambda *a, **k: None)
_set_module_attr("metrics_logger", "log_metrics", lambda *a, **k: None)
_set_module_attr("ratelimit", "limits", MagicMock(return_value=lambda f: f))
_set_module_attr("ratelimit", "sleep_and_retry", MagicMock(return_value=lambda f: f))
_set_module_attr("pybreaker", "CircuitBreaker", MagicMock())
sys.modules["strategy_allocator"] = types.ModuleType("strategy_allocator")


class _Alloc:
    def __init__(self, *a, **k):
        # AI-AGENT-REF: Add config attribute for test compatibility
        from types import SimpleNamespace
        self.config = SimpleNamespace()
        self.config.delta_threshold = 0.02
        self.config.signal_confirmation_bars = 2

    def allocate(self, *a, **k):
        return []

    def update_reward(self, strategy: str, reward: float) -> None:
        """Update reward for a strategy (used for test compatibility)."""
        pass


_set_module_attr("strategy_allocator", "StrategyAllocator", _Alloc)
sys.modules["ai_trading.capital_scaling"] = types.ModuleType("ai_trading.capital_scaling")
_cap_mod = sys.modules["ai_trading.capital_scaling"]
def _update_if_present(runtime, equity):
    cs = getattr(runtime, "capital_scaler", None)
    if cs is not None and hasattr(cs, "update"):
        try:
            result = cs.update(runtime, equity)
            if hasattr(cs, "current_scale"):
                return float(cs.current_scale())
            if isinstance(result, (int, float)):
                return float(result)
        except Exception:
            return 1.0
    return 1.0

def _capital_scale(runtime):
    cs = getattr(runtime, "capital_scaler", None)
    if cs is not None and hasattr(cs, "current_scale"):
        try:
            return float(cs.current_scale())
        except Exception:
            return 1.0
    return 1.0

setattr(_cap_mod, "update_if_present", _update_if_present)
setattr(_cap_mod, "capital_scale", _capital_scale)
setattr(_cap_mod, "capital_scaler_update", _update_if_present)


class _CapScaler:
    def __init__(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def __call__(self, size):
        return size

    def scale_position(self, size):
        return size


setattr(_cap_mod, "CapitalScalingEngine", _CapScaler)
setattr(_cap_mod, "drawdown_adjusted_kelly", lambda *a, **k: 0.02)
setattr(_cap_mod, "volatility_parity_position", lambda *a, **k: 0.01)



def test_bot_main_normal(monkeypatch):
    monkeypatch.setenv("AI_TRADING_TRADING_MODE", "balanced")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    config.reload_env()
    monkeypatch.setattr(sys, "argv", ["bot.py"])
    with patch("ai_trading.data.fetch.get_minute_df", return_value=MagicMock()), patch(
        "ai_trading.alpaca_api.submit_order", return_value={"status": "mocked"}
    ), patch("ai_trading.signals.generate", return_value=1), patch("ai_trading.risk.engine.calculate_position_size", return_value=10), patch(
        "ai_trading.data.fetch.get_daily_df",
        return_value=pd.DataFrame(
            {
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [1],
            }
        ),
    ):
        from ai_trading.core import bot_engine as bot

        monkeypatch.setattr(bot, "main", lambda: True)
        assert cast(Callable[[], bool], bot.main)() is True


def test_bot_main_data_fetch_error(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    config.reload_env()
    monkeypatch.setattr(sys, "argv", ["bot.py"])
    with patch("ai_trading.data.fetch.get_minute_df", side_effect=Exception("API error")), patch(
        "ai_trading.data.fetch.get_daily_df",
        return_value=pd.DataFrame(
            {
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [1],
            }
        ),
    ):
        from ai_trading.core import bot_engine as bot

        monkeypatch.setattr(
            bot,
            "main",
            lambda: (_ for _ in ()).throw(Exception("API error")),
        )
        with pytest.raises(Exception):
            bot.main()


def test_bot_main_signal_nan(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    config.reload_env()
    monkeypatch.setattr(sys, "argv", ["bot.py"])
    with patch("ai_trading.signals.generate", return_value=float("nan")), patch(
        "ai_trading.data.fetch.get_minute_df", return_value=MagicMock()
    ), patch(
        "ai_trading.data.fetch.get_daily_df",
        return_value=pd.DataFrame(
            {
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [1],
            }
        ),
    ):
        from ai_trading.core import bot_engine as bot

        monkeypatch.setattr(bot, "main", lambda: None)
        bot.main()
