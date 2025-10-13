import sys
import types
from types import SimpleNamespace

try:
    import alpaca.trading.requests as alpaca_requests
except ModuleNotFoundError:
    alpaca_root = types.ModuleType("alpaca")
    trading_mod = types.ModuleType("alpaca.trading")
    requests_mod = types.ModuleType("alpaca.trading.requests")
    alpaca_root.trading = trading_mod  # type: ignore[attr-defined]
    trading_mod.requests = requests_mod  # type: ignore[attr-defined]
    sys.modules.setdefault("alpaca", alpaca_root)
    sys.modules.setdefault("alpaca.trading", trading_mod)
    sys.modules.setdefault("alpaca.trading.requests", requests_mod)
    alpaca_requests = requests_mod

if "alpaca.common.exceptions" not in sys.modules:
    exceptions_mod = types.ModuleType("alpaca.common.exceptions")

    class _APIError(Exception):
        pass

    exceptions_mod.APIError = _APIError  # type: ignore[attr-defined]
    sys.modules.setdefault("alpaca.common", types.ModuleType("alpaca.common"))
    sys.modules["alpaca.common.exceptions"] = exceptions_mod

from ai_trading.core import alpaca_client
from ai_trading.execution.live_trading import ExecutionEngine


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
from ai_trading.core import execution_flow


class _Runtime(SimpleNamespace):
    """Lightweight runtime stub that mirrors the required api attribute."""


def test_cancel_all_open_orders_handles_new_and_pending():
    cancelled = []

    class DummyAPI:
        def __init__(self):
            self._orders = [
                SimpleNamespace(id="open", status="open"),
                SimpleNamespace(id="new", status="new"),
                SimpleNamespace(id="pending", status="pending_new"),
                SimpleNamespace(id="other", status="filled"),
            ]

        def list_orders(self, status=None):
            return list(self._orders)

        def cancel_order_by_id(self, order_id):
            cancelled.append(order_id)

    runtime = _Runtime(api=DummyAPI())

    bot_engine.cancel_all_open_orders(runtime)

    assert set(cancelled) == {"open", "new", "pending"}


def test_cancel_all_open_orders_handles_enum_status():
    cancelled = []

    class DummyAPI:
        def list_orders(self, status=None):
            class StatusEnum:
                def __init__(self, value):
                    self.value = value

            return [SimpleNamespace(id="enum", status=StatusEnum("NEW"))]

        def cancel_order_by_id(self, order_id):
            cancelled.append(order_id)

    runtime = _Runtime(api=DummyAPI())

    bot_engine.cancel_all_open_orders(runtime)

    assert cancelled == ["enum"]


def test_cancel_all_open_orders_uses_cancel_orders_request(monkeypatch):
    cancelled_payloads = []

    class DummyAPI:
        def list_orders(self, status=None):
            return [SimpleNamespace(id="target", status="open")]

        def cancel_orders(self, *args, **kwargs):
            if args:
                request = args[0]
            else:
                request = (
                    kwargs.get("request")
                    or kwargs.get("cancel_orders_request")
                )
            if request is None:
                raise TypeError("request required")
            cancelled_payloads.append(getattr(request, "payload", {}))
            return {"status": "ok"}

    class FakeCancelOrdersRequest:
        def __init__(self, **kwargs):
            if not kwargs:
                raise TypeError("payload required")
            self.payload = kwargs

    monkeypatch.setattr(
        alpaca_requests,
        "CancelOrdersRequest",
        FakeCancelOrdersRequest,
        raising=False,
    )

    runtime = _Runtime(api=DummyAPI())

    bot_engine.cancel_all_open_orders(runtime)

    assert cancelled_payloads
    payload = cancelled_payloads[0]
    assert any("target" in value if isinstance(value, str) else "target" in value for value in payload.values())


def test_execution_engine_cancel_order_shim(monkeypatch):
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    class DummyTradingClient:
        def __init__(self):
            self.cancelled = []

        def cancel_order_by_id(self, order_id):
            self.cancelled.append(order_id)

        def list_orders(self, status=None):
            return []

    client = DummyTradingClient()
    alpaca_client._validate_trading_api(client)

    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine.trading_client = client

    assert engine._cancel_order_alpaca("abc123") is True
    assert client.cancelled == ["abc123"]


def test_send_exit_order_uses_cancel_order_shim(monkeypatch):
    class DummyAPI:
        def __init__(self):
            self.cancelled = []
            self.orders = {}

        def list_orders(self, status=None):
            return []

        def get_position(self, symbol):
            return SimpleNamespace(qty=10)

        def get_order(self, order_id):
            return self.orders[order_id]

        def cancel_order_by_id(self, order_id):
            self.cancelled.append(order_id)

    api = DummyAPI()
    alpaca_client._validate_trading_api(api)
    runtime = _Runtime(api=api)

    limit_order_ids = []
    market_calls = []

    def submit_order(**kwargs):
        if not limit_order_ids and kwargs.get("limit_price") is not None:
            order = SimpleNamespace(id="limit-001")
            api.orders[order.id] = SimpleNamespace(id=order.id, status="new")
            limit_order_ids.append(order.id)
            return order
        market_calls.append(kwargs.get("symbol"))
        return SimpleNamespace(id="market-001")

    api.submit_order = submit_order  # type: ignore[attr-defined]

    def fake_safe_submit_order(_api, _req):
        if not limit_order_ids:
            order = SimpleNamespace(id="limit-001")
            api.orders[order.id] = SimpleNamespace(id=order.id, status="new")
            limit_order_ids.append(order.id)
            return order
        market_calls.append(getattr(_req, "symbol", None))
        return SimpleNamespace(id="market-001")

    monkeypatch.setattr(bot_engine, "safe_submit_order", fake_safe_submit_order)
    monkeypatch.setattr(execution_flow, "safe_submit_order", fake_safe_submit_order)
    monkeypatch.setattr(execution_flow.pytime, "sleep", lambda _secs: None)

    execution_flow.send_exit_order(runtime, "AAPL", 5, 150.0, "manual_exit")

    assert api.cancelled == ["limit-001"]
    assert market_calls == ["AAPL"]
