import types
import pytest


class DummyAPI:
    def __init__(self):
        self.get_account = lambda: types.SimpleNamespace(buying_power="1000")
        self.list_positions = lambda: []
        self.submit_order = lambda **_kwargs: types.SimpleNamespace(id=1, status="pending_new", filled_qty=0)
        self.get_order = lambda oid: types.SimpleNamespace(id=1, status="pending_new", filled_qty=0)


def test_safe_submit_order_pending_new(monkeypatch):
    """Test safe_submit_order function with mock dependencies."""

    # Import only after conftest.py has set up mocks
    from ai_trading.core import bot_engine

    # Mock the required functions
    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_alpaca_available", lambda x: True)

    api = DummyAPI()
    req = types.SimpleNamespace(symbol="AAPL", qty=1, side="buy")

    try:
        order = bot_engine.safe_submit_order(api, req)
        assert order is not None
        assert order.status == "pending_new"
        assert isinstance(order.filled_qty, (int, float))
        assert isinstance(order.qty, (int, float))
        assert order.filled_qty == 0
        assert order.qty == 1
    except ImportError:
        pass


def test_safe_submit_order_pending_new_symbol(monkeypatch):
    """Ensure pending-new polling keeps original symbol."""

    from ai_trading.core import bot_engine
    from tests.support.dummy_api import DummyAPI as SymbolDummyAPI

    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_alpaca_available", lambda x: True)

    api = SymbolDummyAPI()
    req = types.SimpleNamespace(symbol="MSFT", qty=1, side="buy")

    order = bot_engine.safe_submit_order(api, req)
    assert order.symbol == "MSFT"
    assert isinstance(order.filled_qty, (int, float))
    assert isinstance(order.qty, (int, float))


class MissingFieldsAPI:
    def __init__(self):
        self.get_account = lambda: types.SimpleNamespace(buying_power="1000")
        self.list_positions = lambda: []
        self.submit_order = lambda **_kwargs: types.SimpleNamespace(id=1, status="filled")
        self.get_order = lambda oid: types.SimpleNamespace(id=1, status="filled")


def test_safe_submit_order_defaults_missing_fields(monkeypatch):
    from ai_trading.core import bot_engine

    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_alpaca_available", lambda x: True)

    api = MissingFieldsAPI()
    req = types.SimpleNamespace(symbol="GOOG", qty=2, side="buy")
    order = bot_engine.safe_submit_order(api, req)
    assert order.qty == 2
    assert order.filled_qty == 0


class NonNumericAPI(DummyAPI):
    def __init__(self) -> None:
        super().__init__()
        self.submit_order = lambda **_kwargs: types.SimpleNamespace(
            id=1, status="filled", filled_qty="abc", qty="xyz"
        )


def test_safe_submit_order_raises_on_non_numeric(monkeypatch):
    from ai_trading.core import bot_engine

    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_alpaca_available", lambda x: True)

    api = NonNumericAPI()
    req = types.SimpleNamespace(symbol="TSLA", qty=1, side="buy")
    with pytest.raises(ValueError):
        bot_engine.safe_submit_order(api, req)


def test_safe_submit_order_generates_id(monkeypatch):
    """safe_submit_order should generate and record a client order ID."""

    from ai_trading.core import bot_engine, order_ids

    monkeypatch.setattr(bot_engine, "market_is_open", lambda: True)
    monkeypatch.setattr(bot_engine, "check_alpaca_available", lambda x: True)

    generated: list[str] = []

    def fake_gen(prefix: str = "ai") -> str:
        cid = f"{prefix}-test"
        generated.append(cid)
        return cid

    monkeypatch.setattr(order_ids, "generate_client_order_id", fake_gen)

    api = DummyAPI()
    submitted: dict[str, dict[str, object]] = {}

    def record_submit_order(**kwargs):
        submitted["args"] = kwargs
        return types.SimpleNamespace(id=1, status="pending_new", filled_qty=0)

    api.submit_order = record_submit_order  # type: ignore[assignment]
    req = types.SimpleNamespace(symbol="IBM", qty=1, side="buy")

    bot_engine.safe_submit_order(api, req)

    assert generated, "ID generator was not called"
    assert api.client_order_ids == generated
    assert submitted["args"].get("client_order_id") == generated[0]
