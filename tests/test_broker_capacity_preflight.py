import pytest

from ai_trading.execution import live_trading as lt
from ai_trading.execution.live_trading import CapacityCheck, ExecutionEngine, preflight_capacity


_REAL_CALL_PREFLIGHT_CAPACITY = lt._call_preflight_capacity


@pytest.fixture(autouse=True)
def _restore_capacity_helper(monkeypatch):
    monkeypatch.setattr(lt, "_call_preflight_capacity", _REAL_CALL_PREFLIGHT_CAPACITY)


class FakeBroker:
    def __init__(self, *, buying_power="0", maintenance_margin="0", non_marginable="0", orders=None):
        self._account = {
            "buying_power": buying_power,
            "maintenance_margin": maintenance_margin,
            "non_marginable_buying_power": non_marginable,
        }
        self._orders = list(orders or [])
        self.get_account_calls = 0
        self.list_orders_calls = 0
        self.submit_calls = 0

    def get_account(self):
        self.get_account_calls += 1
        return self._account

    def list_orders(self, status="open"):
        assert status == "open"
        self.list_orders_calls += 1
        return self._orders

    def submit_order(self, order_data):
        self.submit_calls += 1
        return {"id": "fake", "status": "accepted", "qty": order_data.get("quantity")}


class NoAccountBroker(FakeBroker):
    def get_account(self):
        self.get_account_calls += 1
        raise RuntimeError("account endpoint unavailable")


def test_preflight_capacity_downsizes_quantity(monkeypatch):
    monkeypatch.setenv("EXECUTION_MIN_QTY", "1")
    broker = FakeBroker(
        buying_power="900",
        maintenance_margin="200",
        orders=[
            {"qty": "4", "limit_price": "50", "side": "buy"},
            {"qty": "2", "limit_price": "100", "side": "buy"},
        ],
    )

    check = preflight_capacity("AAPL", "buy", 50, 20, broker)

    assert isinstance(check, CapacityCheck)
    assert check.can_submit is True
    assert check.suggested_qty == 6
    assert check.reason is None
    assert broker.get_account_calls == 1
    assert broker.list_orders_calls == 1


def test_preflight_capacity_rejects_when_below_minimums(monkeypatch):
    monkeypatch.setenv("EXECUTION_MIN_QTY", "5")
    monkeypatch.setenv("EXECUTION_MIN_NOTIONAL", "360")
    broker = FakeBroker(buying_power="350", orders=[])

    check = preflight_capacity("MSFT", "buy", 50, 10, broker)

    assert check.can_submit is False
    assert check.suggested_qty == 7
    assert check.reason == "below_min_notional"


def test_preflight_capacity_fails_closed_when_account_unavailable_in_paper(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    broker = NoAccountBroker(buying_power="0", orders=[])

    check = preflight_capacity("AAPL", "buy", 100, 5, broker)

    assert check.can_submit is False
    assert check.suggested_qty == 0
    assert check.reason == "account_unavailable"


def test_preflight_capacity_fails_closed_in_pytest_when_account_endpoint_unavailable(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    broker = NoAccountBroker(buying_power="0", orders=[])

    check = preflight_capacity("AAPL", "buy", 100, 5, broker)

    assert check.can_submit is False
    assert check.suggested_qty == 0
    assert check.reason == "account_unavailable"


def test_submit_limit_order_skips_when_capacity_fails(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("EXECUTION_MIN_QTY", "1")
    monkeypatch.setenv("EXECUTION_MIN_NOTIONAL", "0")

    engine = ExecutionEngine(execution_mode="paper", shadow_mode=False)
    engine.is_initialized = True

    broker = FakeBroker(buying_power="0", orders=[])
    engine.trading_client = broker

    result = engine.submit_limit_order("TSLA", "buy", 10, limit_price=50)

    assert result is None
    assert broker.submit_calls == 0
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1


def test_submit_limit_order_skips_when_account_cache_is_stale(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("EXECUTION_MIN_QTY", "1")
    monkeypatch.setenv("EXECUTION_MIN_NOTIONAL", "0")

    engine = ExecutionEngine(execution_mode="paper", shadow_mode=False)
    engine.is_initialized = True

    broker = FakeBroker(buying_power="0", orders=[])
    engine.trading_client = broker
    engine._cycle_account = None
    engine._cycle_account_fetched = True

    result = engine.submit_limit_order("TSLA", "buy", 10, limit_price=50)

    assert result is None
    assert broker.submit_calls == 0
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1


def test_submit_limit_order_uses_explicit_broker_when_provider_is_paper(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("EXECUTION_MIN_QTY", "1")
    monkeypatch.setenv("EXECUTION_MIN_NOTIONAL", "0")
    monkeypatch.setenv("AI_TRADING_BROKER_PROVIDER", "paper")

    engine = ExecutionEngine(execution_mode="paper", shadow_mode=False)
    engine.is_initialized = True

    broker = FakeBroker(buying_power="0", orders=[])
    engine.trading_client = broker

    result = engine.submit_limit_order("TSLA", "buy", 10, limit_price=50)

    assert result is None
    assert broker.submit_calls == 0
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1
