"""Regression tests for Alpaca fallback timeframe handling."""

from __future__ import annotations

import importlib
import builtins
import sys
import types
from typing import Any, cast
import pytest


def _clear_module(monkeypatch, prefix: str) -> None:
    for name in [m for m in sys.modules if m.startswith(prefix)]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_missing_sdk_does_not_install_timeframe_fallbacks(monkeypatch, request):
    """Missing alpaca-py must not install fallback request/timeframe classes."""

    original_alpaca = sys.modules.pop("ai_trading.alpaca_api", None)
    if original_alpaca is not None:
        request.addfinalizer(lambda: sys.modules.__setitem__("ai_trading.alpaca_api", original_alpaca))
    else:
        request.addfinalizer(lambda: sys.modules.pop("ai_trading.alpaca_api", None))

    _clear_module(monkeypatch, "alpaca")
    monkeypatch.delitem(sys.modules, "ai_trading.timeframe", raising=False)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("alpaca"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    stub_utils_http = cast(Any, types.ModuleType("ai_trading.utils.http"))
    stub_utils_http.clamp_request_timeout = lambda timeout: timeout
    monkeypatch.setitem(sys.modules, "ai_trading.utils.http", stub_utils_http)

    stub_config_management = cast(Any, types.ModuleType("ai_trading.config.management"))
    stub_config_management.is_shadow_mode = lambda: False
    stub_config_management.get_env = (
        lambda key, default=None, **_kwargs: "1"
        if key == "AI_TRADING_FORCE_ALPACA_UNAVAILABLE"
        else default
    )
    monkeypatch.setitem(sys.modules, "ai_trading.config.management", stub_config_management)

    stub_config_pkg = cast(Any, types.ModuleType("ai_trading.config"))
    stub_config_pkg.management = stub_config_management
    monkeypatch.setitem(sys.modules, "ai_trading.config", stub_config_pkg)

    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    stub_logging = cast(Any, types.ModuleType("ai_trading.logging"))
    stub_logging.get_logger = lambda _name: _Logger()
    monkeypatch.setitem(sys.modules, "ai_trading.logging", stub_logging)

    stub_logging_norm = cast(Any, types.ModuleType("ai_trading.logging.normalize"))
    stub_logging_norm.canon_symbol = lambda sym: str(sym)
    monkeypatch.setitem(sys.modules, "ai_trading.logging.normalize", stub_logging_norm)

    class _Metric:
        def labels(self, *args, **kwargs):  # pragma: no cover - trivial stub
            return self

        def inc(self, *args, **kwargs):  # pragma: no cover - trivial stub
            return None

        def observe(self, *args, **kwargs):  # pragma: no cover - trivial stub
            return None

    stub_metrics = cast(Any, types.ModuleType("ai_trading.metrics"))
    stub_metrics.get_counter = lambda *a, **k: _Metric()
    stub_metrics.get_histogram = lambda *a, **k: _Metric()
    monkeypatch.setitem(sys.modules, "ai_trading.metrics", stub_metrics)

    stub_exc = cast(Any, types.ModuleType("ai_trading.exc"))

    class RequestException(Exception):
        pass

    stub_exc.RequestException = RequestException
    monkeypatch.setitem(sys.modules, "ai_trading.exc", stub_exc)

    stub_net_http = cast(Any, types.ModuleType("ai_trading.net.http"))

    class HTTPSession:  # pragma: no cover - simple stub
        pass

    stub_net_http.HTTPSession = HTTPSession
    stub_net_http.get_http_session = lambda: HTTPSession()
    monkeypatch.setitem(sys.modules, "ai_trading.net.http", stub_net_http)

    alpaca_api = importlib.import_module("ai_trading.alpaca_api")

    assert not alpaca_api.ALPACA_AVAILABLE
    assert alpaca_api.TimeFrame is None
    assert alpaca_api.TimeFrameUnit is None
    assert alpaca_api.StockBarsRequest is None

    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        alpaca_api.get_stock_bars_request_cls()

    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        alpaca_api.get_timeframe_cls()

    timeframe = importlib.import_module("ai_trading.timeframe")
    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        timeframe.TimeFrame()
