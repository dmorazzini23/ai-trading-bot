import logging
import types
import sys

import pytest

cfg_stub = types.ModuleType("ai_trading.config")
cfg_stub.get_settings = lambda: None
sys.modules.setdefault("ai_trading.config", cfg_stub)

utils_stub = types.ModuleType("ai_trading.utils")
utils_stub.__path__ = []  # mark as package
utils_stub.health_check = lambda *a, **k: True
http_stub = types.ModuleType("ai_trading.utils.http")
time_stub = types.ModuleType("ai_trading.utils.time")
time_stub.last_market_session = lambda *a, **k: None
time_stub.now_utc = lambda *a, **k: None
dt_stub = types.ModuleType("ai_trading.utils.datetime")
dt_stub.ensure_datetime = lambda *a, **k: None
sys.modules.setdefault("ai_trading.utils", utils_stub)
sys.modules.setdefault("ai_trading.utils.http", http_stub)
sys.modules.setdefault("ai_trading.utils.time", time_stub)
sys.modules.setdefault("ai_trading.utils.datetime", dt_stub)
alpaca_stub = types.ModuleType("ai_trading.alpaca_api")
alpaca_stub.ALPACA_AVAILABLE = False
alpaca_stub.get_bars_df = lambda *a, **k: None
sys.modules.setdefault("ai_trading.alpaca_api", alpaca_stub)
import ai_trading as _pkg
_pkg.alpaca_api = alpaca_stub

req_stub = types.ModuleType("requests")
exc = types.SimpleNamespace(RequestException=Exception, ConnectionError=Exception, HTTPError=Exception, Timeout=Exception)
req_stub.exceptions = exc
req_stub.get = lambda *a, **k: None
sys.modules.setdefault("requests", req_stub)
sys.modules.setdefault("requests.exceptions", exc)

core_stub = types.ModuleType("ai_trading.core.bot_engine")
class DummyFetcher:
    pass
core_stub.DataFetcher = DummyFetcher
core_stub.DataFetchError = Exception
sys.modules.setdefault("ai_trading.core.bot_engine", core_stub)

import ai_trading.data_fetcher as df


def test_build_fetcher_alpaca(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    alpaca_stub.ALPACA_AVAILABLE = True
    monkeypatch.setitem(sys.modules, "yfinance", None)
    fetcher = df.build_fetcher({})
    assert getattr(fetcher, "source") == "alpaca"


def test_build_fetcher_fallback(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    alpaca_stub.ALPACA_AVAILABLE = False
    monkeypatch.setitem(sys.modules, "yfinance", None)
    fetcher = df.build_fetcher({})
    assert getattr(fetcher, "source") == "fallback"


def test_build_fetcher_singleton(monkeypatch, caplog):
    """Repeated calls should reuse a single DataFetcher instance."""
    df._FETCHER_SINGLETON = None
    alpaca_stub.ALPACA_AVAILABLE = False
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    caplog.set_level("INFO")
    first = df.build_fetcher({})
    second = df.build_fetcher({})
    assert first is second
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("DATA_FETCHER_BUILD")]
    assert len(msgs) == 1


def test_build_fetcher_raises_and_engine_skips(monkeypatch, caplog):
    alpaca_stub.ALPACA_AVAILABLE = False
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(df, "requests", None)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(df.DataFetchError):
        df.build_fetcher({})

    def boom(_cfg):
        raise df.DataFetchError("no fetcher")

    monkeypatch.setattr(df, "build_fetcher", boom)
    logger = logging.getLogger("ai_trading.runner")
    seen = {}

    def fake_warning(msg, *a, **k):
        seen["msg"] = msg

    orig = logger.warning
    logger.warning = fake_warning
    try:
        df.build_fetcher({})
    except df.DataFetchError as e:
        logger.warning("DATA_FETCHER_INIT_FAILED", extra={"detail": str(e)})
    finally:
        logger.warning = orig

    assert seen.get("msg") == "DATA_FETCHER_INIT_FAILED"
