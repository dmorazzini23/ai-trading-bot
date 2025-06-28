import sys, os
sys.path.insert(0, os.getcwd())
print("PYTHONPATH forced to:", sys.path)
from pathlib import Path

import pytest
try:
    import urllib3
except Exception:  # pragma: no cover - optional dependency
    import types
    urllib3 = types.ModuleType("urllib3")
    urllib3.__file__ = "stub"
    sys.modules["urllib3"] = urllib3
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

    # Diagnostic debug prints
    print(f"PYTHONPATH: {sys.path}")
    print(f"urllib3.__file__: {urllib3.__file__}")
    print(f"Has urllib3.util? {hasattr(urllib3, 'util')}")
    try:
        from urllib3.util import Retry
        print(f"urllib3.util.Retry imported successfully: {Retry}")
    except Exception as e:
        print(f"Failed to import urllib3.util.Retry: {e}")

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
    import runner as r
    return importlib.reload(r)
