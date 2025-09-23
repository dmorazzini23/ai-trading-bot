import sys
import types
from types import SimpleNamespace


if "numpy" not in sys.modules:
    class _RandomStub:
        def seed(self, *_args, **_kwargs):
            return None

    class _NumpyStub(types.ModuleType):
        def __init__(self):
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
        def __init__(self, *_args, **_kwargs):
            pass

        def find_all(self, *_args, **_kwargs):
            return []

        def find_parent(self, *_args, **_kwargs):
            return None

        def get_text(self, *_args, **_kwargs):
            return ""

    bs4_stub.BeautifulSoup = lambda *_args, **_kwargs: _Soup()
    sys.modules["bs4"] = bs4_stub


import ai_trading.core.bot_engine as bot_engine


class _Runtime(SimpleNamespace):
    """Lightweight runtime stub that mirrors the required api attribute."""


def test_cancel_all_open_orders_handles_new_and_pending(monkeypatch):
    cancelled = []

    class DummyAPI:
        def cancel_order(self, order_id):
            cancelled.append(order_id)

    runtime = _Runtime(api=DummyAPI())

    orders = [
        SimpleNamespace(id="open", status="open"),
        SimpleNamespace(id="new", status="new"),
        SimpleNamespace(id="pending", status="pending_new"),
        SimpleNamespace(id="other", status="filled"),
    ]

    monkeypatch.setattr(bot_engine, "list_open_orders", lambda api: orders)
    monkeypatch.setattr(bot_engine, "_validate_trading_api", lambda api: True)

    bot_engine.cancel_all_open_orders(runtime)

    assert set(cancelled) == {"open", "new", "pending"}


def test_cancel_all_open_orders_handles_enum_status(monkeypatch):
    cancelled = []

    class DummyAPI:
        def cancel_order(self, order_id):
            cancelled.append(order_id)

    runtime = _Runtime(api=DummyAPI())

    class StatusEnum:
        def __init__(self, value):
            self.value = value

    orders = [SimpleNamespace(id="enum", status=StatusEnum("NEW"))]

    monkeypatch.setattr(bot_engine, "list_open_orders", lambda api: orders)
    monkeypatch.setattr(bot_engine, "_validate_trading_api", lambda api: True)

    bot_engine.cancel_all_open_orders(runtime)

    assert cancelled == ["enum"]
