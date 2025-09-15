"""Regression tests for Alpaca fallback timeframe handling."""

from __future__ import annotations

import importlib
import sys
import types


def _clear_module(monkeypatch, prefix: str) -> None:
    for name in [m for m in sys.modules if m.startswith(prefix)]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_stock_bars_request_accepts_mutable_timeframe(monkeypatch, request):
    """Ensure fallback ``StockBarsRequest`` works with mutable timeframe."""

    original_alpaca = sys.modules.pop("ai_trading.alpaca_api", None)
    if original_alpaca is not None:
        request.addfinalizer(lambda: sys.modules.__setitem__("ai_trading.alpaca_api", original_alpaca))
    else:
        request.addfinalizer(lambda: sys.modules.pop("ai_trading.alpaca_api", None))

    _clear_module(monkeypatch, "alpaca")

    stub_utils_http = types.ModuleType("ai_trading.utils.http")
    stub_utils_http.clamp_request_timeout = lambda timeout: timeout
    monkeypatch.setitem(sys.modules, "ai_trading.utils.http", stub_utils_http)

    stub_config_management = types.ModuleType("ai_trading.config.management")
    stub_config_management.is_shadow_mode = lambda: False
    monkeypatch.setitem(sys.modules, "ai_trading.config.management", stub_config_management)

    stub_config_pkg = types.ModuleType("ai_trading.config")
    stub_config_pkg.management = stub_config_management
    monkeypatch.setitem(sys.modules, "ai_trading.config", stub_config_pkg)

    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    stub_logging = types.ModuleType("ai_trading.logging")
    stub_logging.get_logger = lambda _name: _Logger()
    monkeypatch.setitem(sys.modules, "ai_trading.logging", stub_logging)

    stub_logging_norm = types.ModuleType("ai_trading.logging.normalize")
    stub_logging_norm.canon_symbol = lambda sym: str(sym)
    monkeypatch.setitem(sys.modules, "ai_trading.logging.normalize", stub_logging_norm)

    class _Metric:
        def labels(self, *args, **kwargs):  # pragma: no cover - trivial stub
            return self

        def inc(self, *args, **kwargs):  # pragma: no cover - trivial stub
            return None

        def observe(self, *args, **kwargs):  # pragma: no cover - trivial stub
            return None

    stub_metrics = types.ModuleType("ai_trading.metrics")
    stub_metrics.get_counter = lambda *a, **k: _Metric()
    stub_metrics.get_histogram = lambda *a, **k: _Metric()
    monkeypatch.setitem(sys.modules, "ai_trading.metrics", stub_metrics)

    stub_exc = types.ModuleType("ai_trading.exc")

    class RequestException(Exception):
        pass

    stub_exc.RequestException = RequestException
    monkeypatch.setitem(sys.modules, "ai_trading.exc", stub_exc)

    stub_net_http = types.ModuleType("ai_trading.net.http")

    class HTTPSession:  # pragma: no cover - simple stub
        pass

    stub_net_http.HTTPSession = HTTPSession
    stub_net_http.get_http_session = lambda: HTTPSession()
    monkeypatch.setitem(sys.modules, "ai_trading.net.http", stub_net_http)

    alpaca_api = importlib.import_module("ai_trading.alpaca_api")

    assert not alpaca_api.ALPACA_AVAILABLE

    timeframe = alpaca_api.TimeFrame()
    request = alpaca_api.StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=timeframe,
    )

    # Mutating the timeframe should not raise FrozenInstanceError/AttributeError
    request.timeframe.amount = 5
    request.timeframe.unit = alpaca_api.TimeFrameUnit.Minute

    assert request.timeframe.amount == 5
    assert request.timeframe.unit == alpaca_api.TimeFrameUnit.Minute
