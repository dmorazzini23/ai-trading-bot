import pytest

import sys
from types import SimpleNamespace

from ai_trading.execution import ExecutionEngine
from ai_trading.core.enums import OrderSide


class _QuoteStub:
    def __init__(self, quote_values, guess_values=None, source="alpaca_ask"):
        self.quote_values = iter(quote_values)
        self.guess_values = iter(guess_values if guess_values is not None else quote_values)
        self.last_quote = None
        self.source = source

    def resolve_trade_quote(self, symbol, prefer_backup=False):  # noqa: ARG002
        self.last_quote = next(self.quote_values)
        return SimpleNamespace(price=self.last_quote, source=self.source)

    def get_latest_price(self, symbol, prefer_backup=False):  # noqa: ARG002
        try:
            value = next(self.guess_values)
        except StopIteration:
            value = self.last_quote
        if value is None and self.last_quote is not None:
            return self.last_quote
        return value

    def get_price_source(self, symbol):  # noqa: ARG002
        return self.source


def test_delayed_quote_slippage_flagged(monkeypatch):
    """Large move between quote and fill should trigger slippage alert."""
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "50")
    stub = _QuoteStub([100.0, 100.0], guess_values=[102.0])
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub)
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 50, raising=False)

    def fake_submit(self, order):
        self.orders[order.id] = order
        self.active_orders[order.id] = order
        return SimpleNamespace(id=order.id)

    monkeypatch.setattr(
        "ai_trading.execution.engine.OrderManager.submit_order",
        fake_submit,
        raising=False,
    )
    engine = ExecutionEngine()
    with pytest.raises(AssertionError):
        engine.execute_order("AAPL", OrderSide.BUY, 10)
    order = next(iter(engine.order_manager.orders.values()))
    assert round(order.slippage_bps, 2) > 50


def test_delayed_quote_slippage_within_threshold(monkeypatch):
    """Minor quote movement should record slippage without alert."""
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "50")
    stub = _QuoteStub([100.0, 100.0], guess_values=[100.3])
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub)
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 50, raising=False)

    def fake_submit(self, order):
        self.orders[order.id] = order
        self.active_orders[order.id] = order
        return SimpleNamespace(id=order.id)

    monkeypatch.setattr(
        "ai_trading.execution.engine.OrderManager.submit_order",
        fake_submit,
        raising=False,
    )
    engine = ExecutionEngine()
    order_id = engine.execute_order("AAPL", OrderSide.BUY, 10)
    assert order_id is not None
    order = engine.order_manager.orders[order_id]
    assert round(float(order.expected_price), 2) == 100.0
    assert round(order.slippage_bps, 2) == 30.0
    assert abs(round(order.slippage_bps, 2)) < 50


def test_market_order_uses_backup_price_skips_slippage(monkeypatch):
    """When backup price is used, slippage controls should relax."""
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("MAX_SLIPPAGE_BPS", "50")

    stub = _QuoteStub([101.0], guess_values=[101.0], source="yahoo")
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub)
    monkeypatch.setattr("ai_trading.execution.engine.hash", lambda x: 50, raising=False)

    def fake_submit(self, order):
        self.orders[order.id] = order
        self.active_orders[order.id] = order
        return SimpleNamespace(id=order.id)

    monkeypatch.setattr(
        "ai_trading.execution.engine.OrderManager.submit_order",
        fake_submit,
        raising=False,
    )

    engine = ExecutionEngine()
    order_id = engine.execute_order("AAPL", OrderSide.BUY, 10)
    assert order_id is not None
    order = engine.order_manager.orders[order_id]
    assert getattr(order, "expected_price_source", None) == "yahoo"
    assert order.slippage_bps == 0.0
