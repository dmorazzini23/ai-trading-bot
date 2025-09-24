import sys
import types

import pytest

if "numpy" not in sys.modules:
    class _RandomStub:
        def seed(self, *_args, **_kwargs):
            return None

    class _NumpyStub(types.ModuleType):
        def __init__(self):
            super().__init__("numpy")
            self.random = _RandomStub()
            self.nan = float("nan")

        def __getattr__(self, _name):  # type: ignore[override]
            def _stub(*_args, **_kwargs):
                return 0

            return _stub

    sys.modules["numpy"] = _NumpyStub()

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1
    portalocker_stub.lock = lambda *_args, **_kwargs: None
    portalocker_stub.unlock = lambda *_args, **_kwargs: None
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

from ai_trading.core import bot_engine
from ai_trading.core.order_ids import stable_client_order_id


class FlakyBroker:
    def __init__(self):
        self.attempts = 0
        self.orders: list[types.SimpleNamespace] = []
        self.client_order_ids: list[str] = []
        self.seen_ids: list[str | None] = []

    def get_account(self):
        return types.SimpleNamespace(buying_power="100000")

    def list_positions(self):
        return []

    def submit_order(self, **kwargs):
        self.attempts += 1
        cid = kwargs.get("client_order_id")
        self.seen_ids.append(cid)
        if self.attempts == 1:
            raise TimeoutError("transient timeout")
        order = types.SimpleNamespace(
            id="server-order",
            status="filled",
            qty=kwargs.get("qty", 0),
            filled_qty=kwargs.get("qty", 0),
            symbol=kwargs.get("symbol"),
            client_order_id=cid,
        )
        self.orders.append(order)
        return order

    def get_order(self, order_id):
        if not self.orders:
            raise RuntimeError("no order recorded")
        order = self.orders[-1]
        return types.SimpleNamespace(
            id=order_id,
            status="filled",
            qty=order.qty,
            filled_qty=order.qty,
            symbol=order.symbol,
            client_order_id=order.client_order_id,
        )


@pytest.mark.usefixtures("monkeypatch")
def test_transient_retry_uses_stable_client_id(monkeypatch, caplog):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(bot_engine.random, "uniform", lambda *_: 0.0)
    monkeypatch.setattr(bot_engine.time, "sleep", lambda *_: None)
    fixed_epoch = 1_700_000_000.0
    monkeypatch.setattr(bot_engine.time, "time", lambda: fixed_epoch)

    broker = FlakyBroker()
    request = types.SimpleNamespace(symbol="AAPL", qty=5, side="buy")

    caplog.set_level("INFO")
    order = bot_engine.safe_submit_order(broker, request, bypass_market_check=True)

    expected_id = stable_client_order_id("AAPL", "buy", int(fixed_epoch // 60))
    assert order.client_order_id == expected_id
    assert broker.seen_ids == [expected_id, expected_id]
    assert broker.attempts == 2
    assert len(broker.orders) == 1

    retry_logs = [record for record in caplog.records if record.msg == "ORDER_RETRY_SCHEDULED"]
    assert len(retry_logs) == 1
    retry_record = retry_logs[0]
    assert retry_record.attempt == 2
    assert retry_record.reason == "timeout"
    assert retry_record.symbol == "AAPL"

    gave_up_logs = [record for record in caplog.records if record.msg == "ORDER_RETRY_GAVE_UP"]
    assert not gave_up_logs
