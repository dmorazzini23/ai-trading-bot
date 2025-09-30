"""Regression tests ensuring safe order submission without Alpaca SDK."""

from __future__ import annotations

import importlib
import builtins
import sys
import types
from types import SimpleNamespace

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
        indicators_stub = types.ModuleType("ai_trading.indicators")
        _zero = lambda *args, **kwargs: 0
        indicators_stub.atr = _zero
        indicators_stub.compute_atr = _zero
        indicators_stub.mean_reversion_zscore = _zero
        indicators_stub.rsi = _zero
        indicators_stub.__getattr__ = lambda _name: _zero  # type: ignore[attr-defined]
        sys.modules["ai_trading.indicators"] = indicators_stub

    if "portalocker" not in sys.modules:
        portalocker_stub = types.ModuleType("portalocker")
        portalocker_stub.LOCK_EX = 1
        portalocker_stub.lock = lambda *args, **kwargs: None
        portalocker_stub.unlock = lambda *args, **kwargs: None
        sys.modules["portalocker"] = portalocker_stub

    if "bs4" not in sys.modules:
        bs4_stub = types.ModuleType("bs4")

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


_ensure_numpy_stub()
_ensure_optional_stubs()


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")


def test_ensure_alpaca_classes_assigns_fallbacks(monkeypatch: pytest.MonkeyPatch):
    _block_alpaca_imports(monkeypatch)

    from ai_trading.core import bot_engine

    _reset_alpaca_symbols(bot_engine, monkeypatch)

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
        assert getattr(bot_engine, attr) is not None
    assert bot_engine._ALPACA_IMPORT_ERROR is None
    assert getattr(bot_engine.OrderSide, "BUY", None)
    filled_member = getattr(bot_engine.OrderStatus, "FILLED", None)
    assert filled_member is not None
    assert str(getattr(filled_member, "value", filled_member)).lower() == "filled"


def test_safe_submit_order_without_alpaca(monkeypatch: pytest.MonkeyPatch):
    _block_alpaca_imports(monkeypatch)

    from ai_trading.core import bot_engine

    _reset_alpaca_symbols(bot_engine, monkeypatch)

    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)

    class DummyAPI:
        def __init__(self) -> None:
            self.client_order_ids: list[str] = []
            self._latest: dict[str, object] = {}

        def get_account(self):
            return SimpleNamespace(buying_power="1000")

        def list_positions(self):
            return []

        def submit_order(self, **kwargs):  # type: ignore[no-untyped-def]
            self._latest = kwargs
            return SimpleNamespace(
                id="order-1",
                status="pending_new",
                filled_qty=0,
                qty=kwargs.get("qty", 0),
                symbol=kwargs.get("symbol", ""),
            )

        def get_order(self, order_id):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                id=order_id,
                status="filled",
                filled_qty=self._latest.get("qty", 0),
                qty=self._latest.get("qty", 0),
                symbol=self._latest.get("symbol", ""),
            )

    api = DummyAPI()
    req = SimpleNamespace(symbol="AAPL", qty=1, side="buy", time_in_force="day")

    order = bot_engine.safe_submit_order(api, req)

    assert order is not None
    assert getattr(order, "status", "") == "filled"
    assert getattr(order, "qty", 0) == 1
    assert api.client_order_ids, "client order id should be recorded"


def test_safe_submit_order_without_alpaca_order_data(monkeypatch: pytest.MonkeyPatch):
    _block_alpaca_imports(monkeypatch)

    from ai_trading.core import bot_engine

    _reset_alpaca_symbols(bot_engine, monkeypatch)

    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)

    class OrderDataAPI:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.client_order_ids: list[str] = []

        def get_account(self):
            return SimpleNamespace(buying_power="1000")

        def list_positions(self):
            return [SimpleNamespace(symbol="MSFT", qty="10")]

        def submit_order(self, *, order_data):  # type: ignore[no-untyped-def]
            self.calls.append(order_data)
            return SimpleNamespace(
                id="order-2",
                status="filled",
                filled_qty=getattr(order_data, "qty", 0),
                qty=getattr(order_data, "qty", 0),
                symbol=getattr(order_data, "symbol", ""),
            )

        def get_order(self, order_id):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                id=order_id,
                status="filled",
                filled_qty=getattr(self.calls[-1], "qty", 0) if self.calls else 0,
                qty=getattr(self.calls[-1], "qty", 0) if self.calls else 0,
                symbol=getattr(self.calls[-1], "symbol", "") if self.calls else "",
            )

    api = OrderDataAPI()
    req = SimpleNamespace(symbol="MSFT", qty=3, side="sell", time_in_force="day")

    order = bot_engine.safe_submit_order(api, req)

    assert order is not None
    assert getattr(order, "status", "") == "filled"
    assert api.calls, "submit_order should receive request object"
    sent_request = api.calls[-1]
    assert isinstance(sent_request, bot_engine.MarketOrderRequest)
    assert getattr(sent_request, "symbol", "") == "MSFT"
    assert getattr(sent_request, "qty", 0) == 3


def test_live_trading_request_helper_survives_missing_shims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_alpaca_imports(monkeypatch)

    import ai_trading.execution.live_trading as live_trading

    importlib.reload(live_trading)

    for name in [m for m in list(sys.modules) if m.startswith("ai_trading.alpaca")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setattr(live_trading, "MarketOrderRequest", None, raising=False)
    monkeypatch.setattr(live_trading, "LimitOrderRequest", None, raising=False)
    monkeypatch.setattr(live_trading, "OrderSide", None, raising=False)
    monkeypatch.setattr(live_trading, "TimeInForce", None, raising=False)

    market_cls, limit_cls, side_enum, tif_enum = live_trading._ensure_request_models()

    assert market_cls is not None
    assert limit_cls is not None
    assert side_enum is not None
    assert tif_enum is not None

    order = market_cls(
        symbol="AAPL",
        qty=1,
        side=side_enum.BUY if hasattr(side_enum, "BUY") else "buy",
        time_in_force=tif_enum.DAY if hasattr(tif_enum, "DAY") else "day",
    )
    assert getattr(order, "symbol", "") == "AAPL"
