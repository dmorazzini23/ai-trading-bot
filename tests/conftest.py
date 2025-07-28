import sys
import os

# AI-AGENT-REF: Set test environment variables early to avoid config import errors
os.environ.update({
    "ALPACA_API_KEY": "testkey",
    "ALPACA_SECRET_KEY": "testsecret", 
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    "WEBHOOK_SECRET": "test-webhook-secret",
    "FLASK_PORT": "9000",
    "TESTING": "1"
})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

import pytest
import types

# AI-AGENT-REF: Add numpy stub before any imports that might need it
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    import types
    
    class ArrayStub(list):
        def __init__(self, data=None, dtype=None):
            super().__init__(data or [])
            self.dtype = dtype
            
        def __array__(self):
            return self
            
        def reshape(self, *args):
            return self
            
        def __getattr__(self, name):
            return lambda *args, **kwargs: self
    
    numpy_mod = types.ModuleType("numpy")
    numpy_mod.array = ArrayStub
    numpy_mod.ndarray = ArrayStub
    numpy_mod.nan = float('nan')
    numpy_mod.inf = float('inf')
    numpy_mod.asarray = ArrayStub
    numpy_mod.zeros = lambda *args, **kwargs: ArrayStub([0] * (args[0] if args else 1))
    numpy_mod.ones = lambda *args, **kwargs: ArrayStub([1] * (args[0] if args else 1))
    numpy_mod.mean = lambda x, **kwargs: sum(x) / len(x) if x else 0
    numpy_mod.std = lambda x, **kwargs: 1.0
    numpy_mod.sqrt = lambda x: x ** 0.5
    numpy_mod.sum = sum
    numpy_mod.exp = lambda x: 2.718281828 ** x
    numpy_mod.log = lambda x: 0.0
    numpy_mod.maximum = max
    numpy_mod.minimum = min
    numpy_mod.__file__ = "stub"
    sys.modules["numpy"] = numpy_mod
    sys.modules["np"] = numpy_mod

try:
    import urllib3
except Exception:  # pragma: no cover - optional dependency
    import types
    urllib3 = types.ModuleType("urllib3")
    urllib3.__file__ = "stub"
    sys.modules["urllib3"] = urllib3

# AI-AGENT-REF: Add pandas stub for strategy allocator tests
try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    import types
    
    # Create pandas stub module
    pandas_mod = types.ModuleType("pandas")
    
    # Create minimal DataFrame stub
    class DataFrameStub:
        def __init__(self, data=None, **kwargs):
            self.data = data or {}
            
        def __getitem__(self, key):
            return [1, 2, 3]  # Return dummy data
            
        def iloc(self):
            return self
            
        def __getattr__(self, name):
            return lambda *args, **kwargs: self
    
    # Create minimal Timestamp stub
    class TimestampStub:
        @staticmethod
        def utcnow():
            from datetime import datetime, timezone
            return datetime.now(timezone.utc)
    
    # Add pandas functions
    def read_csv(*args, **kwargs):
        return DataFrameStub()
    
    def read_parquet(*args, **kwargs):
        return DataFrameStub()
    
    def concat(*args, **kwargs):
        return DataFrameStub()

    pandas_mod.DataFrame = DataFrameStub
    pandas_mod.Timestamp = TimestampStub
    pandas_mod.Series = list
    pandas_mod.read_csv = read_csv
    pandas_mod.read_parquet = read_parquet
    pandas_mod.concat = concat
    pandas_mod.__file__ = "stub"
    sys.modules["pandas"] = pandas_mod
    sys.modules["pd"] = pandas_mod
try:
    import requests  # ensure real package available
except Exception:  # pragma: no cover - allow missing in test env
    req_mod = types.ModuleType("requests")
    exc_mod = types.ModuleType("requests.exceptions")
    exc_mod.RequestException = Exception
    req_mod.get = lambda *a, **k: None
    req_mod.Session = lambda *a, **k: None
    req_mod.exceptions = exc_mod
    sys.modules["requests"] = req_mod
    sys.modules["requests.exceptions"] = exc_mod

# AI-AGENT-REF: Add additional dependency stubs for tests
try:
    import pandas_ta
except Exception:  # pragma: no cover - optional dependency
    import types
    ta_mod = types.ModuleType("pandas_ta")
    ta_mod.rsi = lambda *a, **k: [50] * 14  # Return dummy RSI values
    ta_mod.atr = lambda *a, **k: [1.0] * 14  # Return dummy ATR values
    ta_mod.__file__ = "stub"
    sys.modules["pandas_ta"] = ta_mod

try:
    import numba
except Exception:  # pragma: no cover - optional dependency
    import types
    
    def jit_stub(*args, **kwargs):
        """Stub for numba.jit decorator - just returns the function unchanged."""
        if len(args) == 1 and callable(args[0]):
            return args[0]  # Direct decoration
        else:
            return lambda func: func  # Parameterized decoration
    
    numba_mod = types.ModuleType("numba")
    numba_mod.jit = jit_stub
    numba_mod.__file__ = "stub"
    sys.modules["numba"] = numba_mod

try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover - optional dependency
    import types
    
    class BaseSettingsStub:
        def __init__(self, **kwargs):
            # Read from environment variables
            import os
            self.ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "testkey")
            self.ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "testsecret")
            self.ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            self.ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # Missing attribute added
            self.FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", None)
            self.FUNDAMENTAL_API_KEY = os.getenv("FUNDAMENTAL_API_KEY", None)
            self.NEWS_API_KEY = os.getenv("NEWS_API_KEY", None)
            self.IEX_API_TOKEN = os.getenv("IEX_API_TOKEN", None)
            self.WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "test-webhook-secret")
            self.FLASK_PORT = int(os.getenv("FLASK_PORT", "9000"))
            self.BOT_MODE = os.getenv("BOT_MODE", "balanced")
            self.MODEL_PATH = os.getenv("MODEL_PATH", "trained_model.pkl")
            self.HALT_FLAG_PATH = os.getenv("HALT_FLAG_PATH", "halt.flag")
            self.MAX_PORTFOLIO_POSITIONS = int(os.getenv("MAX_PORTFOLIO_POSITIONS", "20"))
            self.LIMIT_ORDER_SLIPPAGE = float(os.getenv("LIMIT_ORDER_SLIPPAGE", "0.005"))
            self.HEALTHCHECK_PORT = int(os.getenv("HEALTHCHECK_PORT", "8081"))
            self.RUN_HEALTHCHECK = os.getenv("RUN_HEALTHCHECK", "0")
            self.BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.5"))
            self.WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "9000"))
            self.SLIPPAGE_THRESHOLD = float(os.getenv("SLIPPAGE_THRESHOLD", "0.003"))
            self.REBALANCE_INTERVAL_MIN = int(os.getenv("REBALANCE_INTERVAL_MIN", "1440"))
            self.SHADOW_MODE = os.getenv("SHADOW_MODE", "False").lower() == "true"
            self.DRY_RUN = os.getenv("DRY_RUN", "False").lower() == "true"
            self.DISABLE_DAILY_RETRAIN = os.getenv("DISABLE_DAILY_RETRAIN", "False").lower() == "true"
            self.TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "data/trades.csv")
            self.FORCE_TRADES = os.getenv("FORCE_TRADES", "False").lower() == "true"
            self.DISASTER_DD_LIMIT = float(os.getenv("DISASTER_DD_LIMIT", "0.2"))
            # Add missing attributes from validate_env.py
            self.MODEL_RF_PATH = os.getenv("MODEL_RF_PATH", "model_rf.pkl")
            self.MODEL_XGB_PATH = os.getenv("MODEL_XGB_PATH", "model_xgb.pkl")
            self.MODEL_LGB_PATH = os.getenv("MODEL_LGB_PATH", "model_lgb.pkl")
            self.RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "rl_agent.zip")
            self.USE_RL_AGENT = os.getenv("USE_RL_AGENT", "False").lower() == "true"
            self.SECTOR_EXPOSURE_CAP = float(os.getenv("SECTOR_EXPOSURE_CAP", "0.4"))
            self.MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
            self.WEEKLY_DRAWDOWN_LIMIT = float(os.getenv("WEEKLY_DRAWDOWN_LIMIT", "0.15"))
            self.VOLUME_THRESHOLD = int(os.getenv("VOLUME_THRESHOLD", "50000"))
            self.DOLLAR_RISK_LIMIT = float(os.getenv("DOLLAR_RISK_LIMIT", "0.02"))
            self.FINNHUB_RPM = int(os.getenv("FINNHUB_RPM", "60"))
            self.MINUTE_CACHE_TTL = int(os.getenv("MINUTE_CACHE_TTL", "60"))
            self.EQUITY_EXPOSURE_CAP = float(os.getenv("EQUITY_EXPOSURE_CAP", "2.5"))
            self.PORTFOLIO_EXPOSURE_CAP = float(os.getenv("PORTFOLIO_EXPOSURE_CAP", "2.5"))
            self.SEED = int(os.getenv("SEED", "42"))
            self.RATE_LIMIT_BUDGET = int(os.getenv("RATE_LIMIT_BUDGET", "190"))
            for k, v in kwargs.items():
                setattr(self, k, v)
                
        @staticmethod
        def model_json_schema():
            return {}
    
    class SettingsConfigDictStub:
        def __init__(self, **kwargs):
            pass
    
    pydantic_settings_mod = types.ModuleType("pydantic_settings")
    pydantic_settings_mod.BaseSettings = BaseSettingsStub
    pydantic_settings_mod.SettingsConfigDict = SettingsConfigDictStub
    pydantic_settings_mod.__file__ = "stub"
    sys.modules["pydantic_settings"] = pydantic_settings_mod

try:
    import pydantic
except Exception:  # pragma: no cover - optional dependency
    import types
    
    class FieldStub:
        def __init__(self, *args, **kwargs):
            pass
            
        def __call__(self, *args, **kwargs):
            return lambda x: x
    
    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.Field = FieldStub()
    pydantic_mod.__file__ = "stub"
    sys.modules["pydantic"] = pydantic_mod

# AI-AGENT-REF: Add alpaca_trade_api stubs
try:
    import alpaca_trade_api
except Exception:  # pragma: no cover - optional dependency
    import types
    
    alpaca_mod = types.ModuleType("alpaca_trade_api")
    rest_mod = types.ModuleType("alpaca_trade_api.rest")
    
    class RESTStub:
        def __init__(self, *args, **kwargs):
            pass
            
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    rest_mod.REST = RESTStub
    alpaca_mod.rest = rest_mod
    alpaca_mod.__file__ = "stub"
    sys.modules["alpaca_trade_api"] = alpaca_mod
    sys.modules["alpaca_trade_api.rest"] = rest_mod

# AI-AGENT-REF: Add other missing dependencies
try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    import types
    psutil_mod = types.ModuleType("psutil")
    psutil_mod.__file__ = "stub"
    sys.modules["psutil"] = psutil_mod

try:
    import tzlocal
except Exception:  # pragma: no cover - optional dependency
    import types
    tzlocal_mod = types.ModuleType("tzlocal")
    tzlocal_mod.get_localzone = lambda: None
    tzlocal_mod.__file__ = "stub"
    sys.modules["tzlocal"] = tzlocal_mod
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*a, **k):
        pass


def pytest_configure() -> None:
    """Load environment variables for tests."""
    env_file = Path('.env.test')
    if not env_file.exists():
        env_file = Path('.env')
    if env_file.exists():
        load_dotenv(env_file)
    # Ensure project root is on the import path so modules like
    # ``ai_trading.capital_scaling`` resolve when tests are run from the ``tests``
    # directory by CI tools or developers.
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    """Provide standard environment variables for tests."""
    monkeypatch.setenv("ALPACA_API_KEY", "testkey")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "testsecret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("FLASK_PORT", "9000")
    monkeypatch.setenv("TESTING", "1")
    yield





import importlib
import types


def reload_module(mod):
    """Reload a module within tests."""
    return importlib.reload(mod)


@pytest.fixture(autouse=True)
def reload_utils_module():
    """Ensure utils is reloaded for each test."""
    import utils
    importlib.reload(utils)
    yield


# AI-AGENT-REF: stub capital scaling helpers for unit tests
@pytest.fixture(autouse=True)
def stub_capital_scaling(monkeypatch):
    """Provide simple stubs for heavy capital scaling functions."""
    import ai_trading.capital_scaling as cs
    monkeypatch.setattr(cs, "drawdown_adjusted_kelly", lambda *a, **k: 0.02)
    monkeypatch.setattr(cs, "volatility_parity_position", lambda *a, **k: 0.01)
    yield


def load_runner(monkeypatch):
    """Import and reload the runner module with a dummy bot."""
    bot_mod = types.ModuleType("bot")
    bot_mod.main = lambda: None
    monkeypatch.setitem(sys.modules, "bot", bot_mod)
    req_mod = types.ModuleType("requests")
    req_mod.get = lambda *a, **k: None
    exc_mod = types.ModuleType("requests.exceptions")
    exc_mod.RequestException = Exception
    req_mod.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "requests.exceptions", exc_mod)
    monkeypatch.setitem(sys.modules, "requests", req_mod)
    alpaca_mod = types.ModuleType("alpaca")
    trading_mod = types.ModuleType("alpaca.trading")
    trading_mod.__path__ = []
    stream_mod = types.ModuleType("alpaca.trading.stream")
    stream_mod.TradingStream = object
    monkeypatch.setitem(sys.modules, "alpaca", alpaca_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.stream", stream_mod)
    import runner as r
    return importlib.reload(r)
