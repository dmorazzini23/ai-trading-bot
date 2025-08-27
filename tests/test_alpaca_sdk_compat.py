import sys
import types

class _NoWait:
    def __init__(self, *a, **k):
        pass

    def __add__(self, other):
        return self

import pytest
pd = pytest.importorskip("pandas")

tenacity_stub = types.ModuleType("tenacity")
class _RetryError(Exception):
    pass
tenacity_stub.RetryError = _RetryError
tenacity_stub.stop_after_attempt = lambda *a, **k: _NoWait()
tenacity_stub.wait_exponential = lambda *a, **k: _NoWait()
tenacity_stub.wait_random = lambda *a, **k: _NoWait()
tenacity_stub.retry_if_exception_type = lambda *a, **k: _NoWait()
sys.modules.setdefault("tenacity", tenacity_stub)
sys.modules.setdefault("portalocker", types.ModuleType("portalocker"))
bs4_stub = types.ModuleType("bs4")
bs4_stub.BeautifulSoup = object
sys.modules.setdefault("bs4", bs4_stub)
flask_stub = types.ModuleType("flask")
class Flask:  # minimal stub
    def __init__(self, *a, **k):
        pass

    def route(self, *a, **k):
        def _decor(fn):
            return fn
        return _decor
flask_stub.Flask = Flask
sys.modules.setdefault("flask", flask_stub)
rebalancer_stub = types.ModuleType("ai_trading.rebalancer")
def maybe_rebalance(*args, **kwargs):  # pragma: no cover - stub
    return None
rebalancer_stub.maybe_rebalance = maybe_rebalance
sys.modules.setdefault("ai_trading.rebalancer", rebalancer_stub)
prom_stub = types.ModuleType("prometheus_client")
prom_stub.REGISTRY = object()
prom_stub.CollectorRegistry = object
class _Noop:
    def __init__(self, *a, **k):
        pass

    def labels(self, *a, **k):
        return self

    def set(self, *a, **k):
        pass

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass
prom_stub.Counter = prom_stub.Gauge = prom_stub.Histogram = prom_stub.Summary = _Noop
prom_stub.start_http_server = lambda *a, **k: None
sys.modules.setdefault("prometheus_client", prom_stub)

from ai_trading.data.bars import (
    StockBarsRequest,
    TimeFrame,
    safe_get_stock_bars,
)
from ai_trading.core.bot_engine import get_stock_bars_safe


def _make_df():
    return pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
            "trade_count": [1],
            "vwap": [1.0],
        },
        index=[pd.Timestamp("2024-01-01", tz="UTC")],
    )


def test_safe_get_stock_bars_uses_get_stock_bars():
    class Client:
        def get_stock_bars(self, request):
            return types.SimpleNamespace(df=_make_df())

    req = StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame.Day)
    df = safe_get_stock_bars(Client(), req, "SPY", "TEST")
    assert not df.empty


def test_safe_get_stock_bars_falls_back_to_get_bars():
    class Client:
        def get_bars(self, symbol_or_symbols, timeframe, **kwargs):
            return _make_df()
    class Req:
        symbol_or_symbols = "SPY"
        timeframe = TimeFrame.Day

    df = safe_get_stock_bars(Client(), Req(), "SPY", "TEST")
    assert not df.empty


def test_get_stock_bars_safe_uses_get_stock_bars():
    class API:
        def get_stock_bars(self, request):  # pragma: no cover - simple stub
            return types.SimpleNamespace(df=_make_df())

    df = get_stock_bars_safe(API(), "SPY", TimeFrame.Day)
    assert not df.empty


def test_get_stock_bars_safe_falls_back_to_get_bars():
    class API:
        def get_bars(self, symbol, timeframe):
            return _make_df()

    df = get_stock_bars_safe(API(), "SPY", TimeFrame.Day)


def test_get_stock_bars_safe_accepts_timeframe_enum(caplog):
    class API:
        def get_stock_bars(self, request):  # pragma: no cover - simple stub
            return types.SimpleNamespace(df=_make_df())

    with caplog.at_level("ERROR"):
        df = get_stock_bars_safe(API(), "SPY", TimeFrame.Day)
    assert not df.empty
    assert caplog.records == []
