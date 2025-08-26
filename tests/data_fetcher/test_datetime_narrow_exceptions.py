import pytest
pd = pytest.importorskip("pandas")
import sys
import types

cfg_stub = types.ModuleType("ai_trading.config")
cfg_stub.get_settings = lambda: None
sys.modules.setdefault("ai_trading.config", cfg_stub)
utils_stub = types.ModuleType("ai_trading.utils")
utils_stub.__path__ = []
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

from ai_trading.data_fetcher import ensure_datetime


def test_dt_invalid_raises_typeerror():
    with pytest.raises(TypeError):
        ensure_datetime(object())


def test_dt_oob_raises_typeerror():
    def bad_ts():
        raise pd.errors.OutOfBoundsDatetime("bad")

    with pytest.raises(TypeError):
        ensure_datetime(bad_ts)
