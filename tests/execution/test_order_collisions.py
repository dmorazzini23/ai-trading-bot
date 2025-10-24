import os
from types import SimpleNamespace

import pytest

from ai_trading.config import get_trading_config
from ai_trading.execution.live_trading import APIError, ExecutionEngine


class _ConflictClient:
    def __init__(self):
        self.cancelled: list[str] = []
        self.submit_calls = 0
        self.list_called = False

    def list_orders(self, status="open", symbols=None):  # noqa: D401 - signature mimic
        self.list_called = True
        return [
            SimpleNamespace(
                id="order-1",
                symbol=(symbols[0] if symbols else "ABBV"),
                side="sell",
                status="accepted",
            )
        ]

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    def get_order(self, order_id):
        return SimpleNamespace(id=order_id, status="canceled")

    def submit_order(self, order_data=None, **kwargs):
        self.submit_calls += 1
        if self.submit_calls == 1:
            raise APIError(
                "cannot open a long buy while a short sell order is open",
                code="40310000",
                status_code=403,
            )
        payload = order_data or kwargs
        return {
            "id": "test-order",
            "status": "accepted",
            "symbol": payload.get("symbol"),
            "client_order_id": payload.get("client_order_id"),
            "qty": payload.get("quantity") or payload.get("qty"),
            "limit_price": payload.get("limit_price"),
        }


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("ORDER_FLIP_MODE", "cancel_then_submit")
    get_trading_config.cache_clear()
    yield
    os.environ.pop("PYTEST_RUNNING", None)


def test_enforce_opposite_policy_cancels_orders(monkeypatch):
    engine = ExecutionEngine(execution_mode="paper", shadow_mode=False)
    client = _ConflictClient()
    engine.trading_client = client

    def _cancel(self, order_id):
        self.trading_client.cancel_order(order_id)
        return True

    monkeypatch.setattr(ExecutionEngine, "_cancel_order_alpaca", _cancel, raising=False)
    allowed, payload = engine._enforce_opposite_side_policy(
        "ABBV",
        "buy",
        10,
        closing_position=False,
        client_order_id="test",
    )
    assert allowed is True
    assert payload is None
    assert client.list_called is True
    assert client.cancelled == ["order-1"]


def test_precheck_skips_when_policy_skip(monkeypatch):
    monkeypatch.setenv("ORDER_FLIP_MODE", "skip")
    get_trading_config.cache_clear()
    engine = ExecutionEngine(execution_mode="paper", shadow_mode=False)
    engine.trading_client = _ConflictClient()
    order = {"symbol": "MSFT", "side": "buy", "quantity": 5}
    allowed = engine._pre_execution_order_checks(order)
    assert allowed is False


def test_is_opposite_conflict_error_detects_conflict():
    class DummyError:
        def __init__(self):
            self.code = "40310000"
            self.status_code = 403
            self.message = "cannot open a long buy while a short sell order is open"

        def __str__(self):
            return self.message

    engine = ExecutionEngine(execution_mode="paper", shadow_mode=False)
    assert engine._is_opposite_conflict_error(DummyError()) is True
