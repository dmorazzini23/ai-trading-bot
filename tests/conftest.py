import sys
import os
from pathlib import Path

import pytest
import types
try:
    import urllib3
except Exception:  # pragma: no cover - optional dependency
    import types
    urllib3 = types.ModuleType("urllib3")
    urllib3.__file__ = "stub"
    sys.modules["urllib3"] = urllib3
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
    # ``capital_scaling`` resolve when tests are run from the ``tests``
    # directory by CI tools or developers.
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    os.environ.setdefault("ALPACA_API_KEY", "testkey")
    os.environ.setdefault("ALPACA_SECRET_KEY", "testsecret")
    os.environ.setdefault("FLASK_PORT", "9000")
    os.environ.setdefault("TESTING", "1")



@pytest.fixture(autouse=True, scope="session")
def cleanup_test_env_vars():
    """Clean up testing flag after session."""
    yield
    os.environ.pop("TESTING", None)


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
