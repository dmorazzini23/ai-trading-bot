import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from ai_trading.config import management as config

pd = pytest.importorskip("pandas")

"""Minimal import-time stubs so strategy_allocator and other modules load."""
try:
    pass  # type: ignore
except ImportError:
    sys.modules["pandas"] = types.ModuleType("pandas")
    sys.modules["pandas"].DataFrame = MagicMock()
    sys.modules["pandas"].Series = MagicMock()
    sys.modules["pandas"].concat = MagicMock()

pytestmark = pytest.mark.integration

try:
    import numpy  # type: ignore  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")
    sys.modules["numpy"].array = MagicMock()
    sys.modules["numpy"].nan = float("nan")
    sys.modules["numpy"].NaN = float("nan")
    sys.modules["numpy"].random = MagicMock()
    sys.modules["numpy"].arange = MagicMock()

try:
    import pandas_ta  # type: ignore  # noqa: F401
except ImportError:
    sys.modules["pandas_ta"] = types.ModuleType("pandas_ta")
if "pandas_ta" in sys.modules:
    mod = sys.modules["pandas_ta"]
    if not hasattr(mod, "momentum"):
        mod.momentum = types.SimpleNamespace(rsi=MagicMock())
    mod.atr = getattr(mod, "atr", MagicMock(return_value=MagicMock()))
    mod.rsi = getattr(mod, "rsi", MagicMock(return_value=MagicMock()))
    mod.macd = getattr(mod, "macd", MagicMock(return_value={"MACD_12_26_9": MagicMock()}))
    mod.sma = getattr(mod, "sma", MagicMock(return_value=MagicMock()))

try:
    import pandas_market_calendars  # type: ignore  # noqa: F401
except ImportError:
    sys.modules["pandas_market_calendars"] = types.ModuleType("pandas_market_calendars")
if not hasattr(sys.modules["pandas_market_calendars"], "get_calendar"):
    sys.modules["pandas_market_calendars"].get_calendar = MagicMock()

mods = [
    "tzlocal",  # AI-AGENT-REF: pytz no longer required
    "requests",
    "urllib3",
    "bs4",
    "flask",
    "schedule",
    "portalocker",
    "alpaca",
    "alpaca.trading.client",
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
    "torch",
]
for m in mods:
    sys.modules.setdefault(m, types.ModuleType(m))

if "torch" in sys.modules:
    sys.modules["torch"].manual_seed = lambda *a, **k: None
    sys.modules["torch"].Tensor = object
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.Module = object
    torch_nn.Sequential = lambda *a, **k: None
    torch_nn.Linear = lambda *a, **k: None
    torch_nn.ReLU = lambda *a, **k: None
    torch_nn.Softmax = lambda *a, **k: None
    sys.modules["torch.nn"] = torch_nn

req_mod = types.ModuleType("requests")
sys.modules["requests"] = req_mod
exc_mod = types.ModuleType("requests.exceptions")
exc_mod.RequestException = Exception
exc_mod.HTTPError = Exception
req_mod.exceptions = exc_mod
req_mod.get = lambda *a, **k: None
req_mod.post = lambda *a, **k: None
req_mod.RequestException = Exception
sys.modules["requests.exceptions"] = exc_mod
sys.modules["urllib3"] = types.ModuleType("urllib3")
sys.modules["urllib3"].exceptions = types.SimpleNamespace(HTTPError=Exception)
sys.modules["alpaca"].TradingClient = object
sys.modules["alpaca"].APIError = Exception
sys.modules["alpaca.trading.client"] = types.ModuleType("alpaca.trading.client")
sys.modules["alpaca.trading.client"].TradingClient = object
sys.modules["alpaca.trading.client"].APIError = Exception


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

    def __init__(self, *a, **k):
        pass


class _TFUnit:
    Minute = "Minute"
    Hour = "Hour"
    Day = "Day"




class _FClient:
    def __init__(self, *a, **k):
        pass


sys.modules["finnhub"].Client = _FClient
sys.modules["finnhub"].FinnhubAPIException = Exception
sys.modules["bs4"].BeautifulSoup = lambda *a, **k: None


class _Flask:
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def decorator(f):
            return f

        return decorator


sys.modules["flask"].Flask = _Flask


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


sys.modules["sklearn.ensemble"].RandomForestClassifier = _RFC
sys.modules["sklearn.linear_model"].Ridge = _Ridge
sys.modules["sklearn.linear_model"].BayesianRidge = _BR
sys.modules["sklearn.decomposition"].PCA = _PCA
sys.modules["prometheus_client"].start_http_server = lambda *a, **k: None
sys.modules["prometheus_client"].Counter = lambda *a, **k: None
sys.modules["prometheus_client"].Gauge = lambda *a, **k: None
sys.modules["prometheus_client"].Histogram = lambda *a, **k: None
sys.modules["pipeline"] = types.ModuleType("pipeline")
sys.modules["metrics_logger"] = types.ModuleType("metrics_logger")
sys.modules["pybreaker"] = types.ModuleType("pybreaker")
sys.modules["pipeline"].model_pipeline = lambda *a, **k: None
sys.modules["metrics_logger"].log_metrics = lambda *a, **k: None
sys.modules["ratelimit"].limits = MagicMock(return_value=lambda f: f)
sys.modules["ratelimit"].sleep_and_retry = MagicMock(return_value=lambda f: f)
sys.modules["pybreaker"].CircuitBreaker = MagicMock()
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


sys.modules["strategy_allocator"].StrategyAllocator = _Alloc
sys.modules["ai_trading.capital_scaling"] = types.ModuleType("ai_trading.capital_scaling")


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
sys.modules["ai_trading.capital_scaling"].drawdown_adjusted_kelly = lambda *a, **k: 0.02
sys.modules["ai_trading.capital_scaling"].volatility_parity_position = lambda *a, **k: 0.01



def test_bot_main_normal(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "shadow")
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

        setattr(bot, "main", lambda: True)

        monkeypatch.setattr(bot, "main", lambda: True)
        assert bot.main() is True


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


