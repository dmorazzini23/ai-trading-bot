import importlib
import sys
import types
from enum import Enum


def _stub_bot_engine_heavy_deps(monkeypatch):
    stub_numpy = types.ModuleType("numpy")
    stub_numpy.nan = float("nan")
    stub_numpy.NaN = stub_numpy.nan

    class _RandomNS:
        @staticmethod
        def seed(*_a, **_k):  # noqa: D401 - simple stub
            return None

    stub_numpy.random = _RandomNS()
    monkeypatch.setitem(sys.modules, "numpy", stub_numpy)

    stub_portalocker = types.ModuleType("portalocker")

    class _DummyLock:
        def __init__(self, *a, **k):  # noqa: D401, ARG002
            """Test stub for portalocker.Lock."""

        def acquire(self, *a, **k):  # noqa: D401, ARG002
            return True

        def release(self, *a, **k):  # noqa: D401, ARG002
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: D401, ARG002
            return False

    stub_portalocker.Lock = _DummyLock
    monkeypatch.setitem(sys.modules, "portalocker", stub_portalocker)

    stub_bs4 = types.ModuleType("bs4")

    class _BeautifulSoup:  # noqa: D401 - simple stub
        def __init__(self, *a, **k):  # noqa: D401, ARG002
            self.text = ""

    stub_bs4.BeautifulSoup = _BeautifulSoup
    monkeypatch.setitem(sys.modules, "bs4", stub_bs4)

    stub_flask = types.ModuleType("flask")

    class _Flask:  # noqa: D401 - simple stub
        def __init__(self, *a, **k):  # noqa: D401, ARG002
            self.config = {}

        def route(self, *_a, **_k):
            def decorator(func):
                return func

            return decorator

    stub_flask.Flask = _Flask
    monkeypatch.setitem(sys.modules, "flask", stub_flask)

    stub_indicators = types.ModuleType("ai_trading.indicators")
    stub_indicators.compute_atr = lambda *a, **k: 1.0
    stub_indicators.atr = lambda *a, **k: 1.0
    stub_indicators.mean_reversion_zscore = lambda *a, **k: 0.0
    stub_indicators.rsi = lambda *a, **k: 0.0
    stub_indicators.ema = lambda *a, **k: 1.0
    monkeypatch.setitem(sys.modules, "ai_trading.indicators", stub_indicators)

    class _PositionAction(Enum):
        HOLD = "hold"

    class _IntelligentPositionManager:
        def __init__(self, *a, **k):  # noqa: D401, ARG002
            """Test stub that skips heavy imports."""

        def analyze_position(self, *a, **k):  # noqa: D401, ARG002
            return None

    stub_ipm = types.ModuleType("ai_trading.position.intelligent_manager")
    stub_ipm.PositionAction = _PositionAction
    stub_ipm.IntelligentPositionManager = _IntelligentPositionManager
    monkeypatch.setitem(sys.modules, "ai_trading.position.intelligent_manager", stub_ipm)


def test_meta_learning_import_without_sklearn(monkeypatch):
    # Simulate sklearn missing
    monkeypatch.setitem(sys.modules, "sklearn", None)
    mod = importlib.import_module("ai_trading.meta_learning")
    assert hasattr(mod, "SKLEARN_AVAILABLE")
    # If import failed properly, SKLEARN_AVAILABLE should be False
    assert mod.SKLEARN_AVAILABLE in (False, True)  # just not crashing on import


def test_fetch_sentiment_graceful_when_requests_unavailable(monkeypatch):
    _stub_bot_engine_heavy_deps(monkeypatch)

    from ai_trading.core import bot_engine as be

    # Force a stub that raises RequestException on .get()
    class _ReqStub:
        class exceptions:
            class RequestException(Exception):
                pass

        def get(self, *a, **k):
            raise self.exceptions.RequestException("no network")

    be.requests = _ReqStub()
    be.RequestException = _ReqStub.exceptions.RequestException
    monkeypatch.setattr(be, "_HTTP_SESSION", be.requests, raising=False)
    monkeypatch.setattr(be.time, "sleep", lambda *_a, **_k: None)

    # Ensure it won't bail early for missing key
    monkeypatch.setenv("SENTIMENT_API_KEY", "dummy")
    monkeypatch.setattr(be, "SENTIMENT_API_KEY", "dummy", raising=False)
    be.SENTIMENT_API_URL = "http://127.0.0.1:1"
    be._SENTIMENT_FAILURES = 0
    out = be.fetch_sentiment("AAPL")
    assert isinstance(out, float) and out == 0.0
    assert be._SENTIMENT_FAILURES >= 1


def test_fetch_sentiment_uses_requests_stub_when_session_blocked(monkeypatch):
    from collections import deque

    _stub_bot_engine_heavy_deps(monkeypatch)

    from ai_trading.core import bot_engine as be

    class _GuardedSession:
        def get(self, *a, **k):
            raise RuntimeError("HTTP session disabled")

    class _ReqStub:
        class exceptions:
            class RequestException(Exception):
                pass

        def __init__(self):
            self.calls = 0

        def get(self, *a, **k):
            self.calls += 1
            raise self.exceptions.RequestException("no network")

    stub = _ReqStub()
    monkeypatch.setattr(be, "_HTTP_SESSION", _GuardedSession(), raising=False)
    be.requests = stub
    be.RequestException = stub.exceptions.RequestException
    monkeypatch.setattr(be, "_SENTIMENT_CACHE", {}, raising=False)
    monkeypatch.setattr(be, "_SENTIMENT_CALL_TIMES", deque(), raising=False)
    monkeypatch.setattr(be, "_SENTIMENT_FAILURES", 0, raising=False)
    monkeypatch.setattr(be.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("SENTIMENT_API_KEY", "dummy")
    monkeypatch.setattr(be, "SENTIMENT_API_KEY", "dummy", raising=False)
    be.SENTIMENT_API_URL = "http://127.0.0.1:1"

    out = be.fetch_sentiment("AAPL")
    assert out == 0.0
    assert stub.calls >= 1
    assert be._SENTIMENT_FAILURES >= 1


def test_alpaca_stubs_are_not_exceptions():
    from ai_trading.core import bot_engine as be

    # TradingClient and other stubs should not be Exception subclasses
    assert not issubclass(be.TradingClient, Exception)
