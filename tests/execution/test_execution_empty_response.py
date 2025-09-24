import types

import pytest

from ai_trading.execution import live_trading


class _DummyTradingClient:
    def __init__(self, response):
        self._response = response
        self.submitted: list[object] = []

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return self._response


class _DummyLimitOrderRequest:
    def __init__(self, *, limit_price, **kwargs):
        self.limit_price = limit_price
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyMarketOrderRequest:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture(autouse=True)
def patch_alpaca_classes(monkeypatch):
    monkeypatch.setattr(live_trading, "OrderSide", types.SimpleNamespace(BUY="buy", SELL="sell"))
    monkeypatch.setattr(live_trading, "TimeInForce", types.SimpleNamespace(DAY="day"))
    monkeypatch.setattr(live_trading, "LimitOrderRequest", _DummyLimitOrderRequest)
    monkeypatch.setattr(live_trading, "MarketOrderRequest", _DummyMarketOrderRequest)
    yield


def _make_engine(monkeypatch, response):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.stats = {
        "total_execution_time": 0.0,
        "total_orders": 0,
        "successful_orders": 0,
        "failed_orders": 0,
        "retry_count": 0,
        "capacity_skips": 0,
        "skipped_orders": 0,
        "circuit_breaker_trips": 0,
    }
    engine.retry_config = {
        "base_delay": 0.01,
        "max_attempts": 1,
        "exponential_base": 2,
        "max_delay": 0.01,
    }
    engine.circuit_breaker = {
        "failure_count": 0,
        "max_failures": 5,
        "reset_time": 300,
        "last_failure": None,
        "is_open": False,
    }
    engine.trading_client = _DummyTradingClient(response)
    monkeypatch.setattr(engine, "_refresh_settings", lambda: None)
    monkeypatch.setattr(engine, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(engine, "_pre_execution_checks", lambda: True)
    monkeypatch.setattr(
        engine,
        "_execute_with_retry",
        types.MethodType(lambda self, func, *args, **kwargs: func(*args, **kwargs), engine),
    )
    return engine


def test_submit_limit_order_handles_empty_response(monkeypatch):
    engine = _make_engine(monkeypatch, response=None)
    result = engine.submit_limit_order("AAPL", "buy", 5, limit_price=123.45)

    assert result["symbol"] == "AAPL"
    assert result["qty"] == 5
    assert result["status"] == "accepted"
    assert result["id"] == result["client_order_id"]
    assert result["id"].startswith("alpaca-") or result["id"].startswith("order_")


def test_execute_order_uses_fallback_identifier(monkeypatch):
    fallback = {
        "id": "",
        "client_order_id": "client-123",
        "status": None,
        "symbol": "AAPL",
        "qty": 7,
        "limit_price": 99.5,
        "raw": None,
    }

    engine = _make_engine(monkeypatch, response=fallback)
    result = engine.execute_order("AAPL", "buy", 7, order_type="limit", limit_price=99.5)

    assert str(result) == "client-123"
    assert result.order.id == "client-123"
    assert result.requested_quantity == 7
