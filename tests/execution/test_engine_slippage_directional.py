import sys
from types import SimpleNamespace

import pytest

if "cachetools" not in sys.modules:
    class _FakeTTLCache(dict):
        def __init__(self, maxsize=0, ttl=0, **kwargs):
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl

    sys.modules["cachetools"] = SimpleNamespace(TTLCache=_FakeTTLCache)

import ai_trading.execution.engine as engine_module
from ai_trading.execution.engine import ExecutionEngine, Order, OrderSide, OrderType


@pytest.fixture(autouse=True)
def _slippage_threshold_env(monkeypatch):
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "5")


@pytest.fixture
def execution_engine(monkeypatch):
    monkeypatch.setattr(engine_module.time, "sleep", lambda *_: None)
    return ExecutionEngine()


def _make_manual_order(side: OrderSide) -> Order:
    return Order(
        "AAPL",
        side,
        100,
        order_type=OrderType.MARKET,
        price=100.0,
        expected_price=100.0,
    )


def test_buy_improvement_slippage_allows_market_execution(execution_engine, monkeypatch):
    monkeypatch.setattr(engine_module, "hash", lambda _: 20, raising=False)
    order = _make_manual_order(OrderSide.BUY)

    execution_engine._simulate_market_execution(order)

    assert order.order_type == OrderType.MARKET
    assert order.quantity == 100


def test_sell_improvement_slippage_allows_market_execution(execution_engine, monkeypatch):
    monkeypatch.setattr(engine_module, "hash", lambda _: 80, raising=False)
    order = _make_manual_order(OrderSide.SELL)

    execution_engine._simulate_market_execution(order)

    assert order.order_type == OrderType.MARKET
    assert order.quantity == 100


@pytest.mark.parametrize(
    "side,hash_value",
    [
        (OrderSide.BUY, 80),
        (OrderSide.SELL, 20),
    ],
)
def test_adverse_slippage_still_raises_for_manual_prices(
    execution_engine, monkeypatch, side, hash_value
):
    monkeypatch.setattr(engine_module, "hash", lambda _: hash_value, raising=False)
    order = _make_manual_order(side)

    with pytest.raises(AssertionError):
        execution_engine._simulate_market_execution(order)
