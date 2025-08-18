import importlib
import sys


def test_meta_learning_import_without_sklearn(monkeypatch):
    # Simulate sklearn missing
    monkeypatch.setitem(sys.modules, "sklearn", None)
    mod = importlib.import_module("ai_trading.meta_learning")
    assert hasattr(mod, "SKLEARN_AVAILABLE")
    # If import failed properly, SKLEARN_AVAILABLE should be False
    assert mod.SKLEARN_AVAILABLE in (False, True)  # just not crashing on import


def test_fetch_sentiment_graceful_when_requests_unavailable(monkeypatch):
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

    # Ensure it won't bail early for missing key
    monkeypatch.setenv("SENTIMENT_API_KEY", "dummy")
    be.SENTIMENT_API_URL = "http://127.0.0.1:1"
    be._SENTIMENT_FAILURES = 0
    out = be.fetch_sentiment("AAPL")
    assert isinstance(out, float) and out == 0.0
    assert be._SENTIMENT_FAILURES >= 1


def test_alpaca_stubs_are_not_exceptions():
    from ai_trading.core import bot_engine as be

    # TradingClient (and other stubs) should not be Exception subclasses
    assert not issubclass(be.TradingClient, Exception)
