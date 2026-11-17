"""Regression tests for no-shim behavior when the Alpaca SDK is unavailable."""

from __future__ import annotations

import importlib
import builtins
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest


def _ensure_numpy_stub() -> None:
    if "numpy" in sys.modules:
        return

    class _RandomStub:
        def seed(self, *_args, **_kwargs):
            return None

    class _NumpyStub(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("numpy")
            self.random = _RandomStub()
            self.nan = float("nan")
            self.NaN = self.nan
            self.ndarray = object

        def __getattr__(self, _name):  # type: ignore[override]
            def _stub(*_args, **_kwargs):
                return 0

            return _stub

    sys.modules["numpy"] = _NumpyStub()


def _ensure_optional_stubs() -> None:
    if "ai_trading.indicators" not in sys.modules:
        indicators_stub = cast(Any, types.ModuleType("ai_trading.indicators"))
        _zero = lambda *args, **kwargs: 0
        indicators_stub.atr = _zero
        indicators_stub.compute_atr = _zero
        indicators_stub.mean_reversion_zscore = _zero
        indicators_stub.rsi = _zero
        indicators_stub.__getattr__ = lambda _name: _zero
        sys.modules["ai_trading.indicators"] = indicators_stub

    if "portalocker" not in sys.modules:
        portalocker_stub = cast(Any, types.ModuleType("portalocker"))
        portalocker_stub.LOCK_EX = 1
        portalocker_stub.lock = lambda *args, **kwargs: None
        portalocker_stub.unlock = lambda *args, **kwargs: None
        sys.modules["portalocker"] = portalocker_stub

    if "bs4" not in sys.modules:
        bs4_stub = cast(Any, types.ModuleType("bs4"))

        class _Soup:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def find_all(self, *_args, **_kwargs):
                return []

            def find_parent(self, *_args, **_kwargs):
                return None

            def get_text(self, *_args, **_kwargs):
                return ""

        bs4_stub.BeautifulSoup = lambda *_args, **_kwargs: _Soup()
        sys.modules["bs4"] = bs4_stub


def _block_alpaca_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        if name.startswith("alpaca"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for module_name in [m for m in list(sys.modules) if m.startswith("alpaca")]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    import ai_trading.alpaca_api as alpaca_api

    importlib.reload(alpaca_api)
    monkeypatch.setattr(alpaca_api, "ALPACA_AVAILABLE", False, raising=False)


def _reset_alpaca_symbols(bot_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    for attr in (
        "Quote",
        "Order",
        "OrderSide",
        "OrderStatus",
        "TimeInForce",
        "MarketOrderRequest",
        "LimitOrderRequest",
        "StopOrderRequest",
        "StopLimitOrderRequest",
        "StockLatestQuoteRequest",
    ):
        monkeypatch.setattr(bot_engine, attr, None, raising=False)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(bot_engine, "ALPACA_AVAILABLE", False, raising=False)


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")


def test_ensure_alpaca_classes_fails_without_sdk(monkeypatch: pytest.MonkeyPatch):
    _block_alpaca_imports(monkeypatch)

    from ai_trading.core import bot_engine

    _reset_alpaca_symbols(bot_engine, monkeypatch)

    with pytest.raises(ModuleNotFoundError):
        bot_engine._ensure_alpaca_classes()

    for attr in (
        "Quote",
        "Order",
        "OrderSide",
        "OrderStatus",
        "TimeInForce",
        "MarketOrderRequest",
        "LimitOrderRequest",
        "StopOrderRequest",
        "StopLimitOrderRequest",
        "StockLatestQuoteRequest",
    ):
        assert getattr(bot_engine, attr) is None






def test_live_trading_request_helper_fails_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_alpaca_imports(monkeypatch)

    monkeypatch.delitem(sys.modules, "ai_trading.execution.live_trading", raising=False)
    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        import ai_trading.execution.live_trading  # noqa: F401
