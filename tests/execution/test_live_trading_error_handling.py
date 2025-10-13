from decimal import Decimal
from typing import Any
from types import SimpleNamespace

import pytest


import ai_trading.execution.live_trading as lt


@pytest.fixture
def engine_factory(monkeypatch):
    """Provide a minimally wired execution engine for limit order tests."""

    monkeypatch.setattr(lt, "_safe_mode_guard", lambda *_, **__: False)

    def _capacity_stub(symbol, side, price_hint, quantity, broker, account_snapshot, preflight_fn=None):
        return lt.CapacityCheck(True, int(quantity))

    monkeypatch.setattr(lt, "_call_preflight_capacity", _capacity_stub)
    monkeypatch.setattr(lt, "get_tick_size", lambda symbol: Decimal("0.01"))

    def _build_engine(execute_behavior):
        engine = object.__new__(lt.ExecutionEngine)
        engine._refresh_settings = lambda: None
        engine.is_initialized = True
        engine._ensure_initialized = lambda: True
        engine._pre_execution_checks = lambda: True
        engine._get_account_snapshot = lambda: None
        engine._should_skip_for_pdt = lambda _account, _closing: (False, None, {})
        engine._execute_with_retry = lambda submit_fn, order_data: execute_behavior(order_data)
        engine._submit_order_to_alpaca = lambda order_data: {"id": "abc123", "order_data": order_data}
        engine.shadow_mode = False
        engine.trading_client = SimpleNamespace()
        engine.stats = {
            "total_execution_time": 0.0,
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
        }
        return engine

    return _build_engine


def test_submit_limit_order_success_path(engine_factory):
    engine = engine_factory(lambda order_data: {"id": "ok", **order_data})

    result = engine.submit_limit_order("AAPL", "buy", 10, 123.45)

    assert result["id"] == "ok"
    assert engine.stats["successful_orders"] == 1
    assert engine.stats["failed_orders"] == 0


def test_submit_limit_order_nonretryable_logs_detail(engine_factory, caplog):
    detail_message = "broker rejected"

    def _raise_nonretryable(_order_data):
        raise lt.NonRetryableBrokerError("rejected", code="R1", detail=detail_message)

    engine = engine_factory(_raise_nonretryable)

    with caplog.at_level("DEBUG", logger=lt.logger.name):
        result = engine.submit_limit_order("AAPL", "buy", 5, 101.0)

    assert result is None
    info_records = [record for record in caplog.records if record.msg == "ORDER_SKIPPED_NONRETRYABLE"]
    assert info_records, "Expected ORDER_SKIPPED_NONRETRYABLE log entry"
    debug_records = [record for record in caplog.records if record.msg == "ORDER_SKIPPED_NONRETRYABLE_DETAIL"]
    assert debug_records, "Expected ORDER_SKIPPED_NONRETRYABLE_DETAIL log entry"
    assert debug_records[0].detail == detail_message


def test_submit_limit_order_unexpected_exception_propagates(engine_factory):
    engine = engine_factory(lambda _order_data: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        engine.submit_limit_order("AAPL", "buy", 2, 55.0)


def test_execute_order_uses_limit_with_fallback(engine_factory, monkeypatch):
    monkeypatch.setenv("EXECUTION_FALLBACK_LIMIT_BUFFER_BPS", "100")

    captured: dict[str, Any] = {}

    def _capture(order_data: dict[str, Any]) -> dict[str, Any]:
        captured["order_data"] = dict(order_data)
        return {"id": "ok", **order_data}

    engine = engine_factory(_capture)

    engine.execute_order(
        "AAPL",
        "buy",
        5,
        price=100.0,
        annotations={"using_fallback_price": True},
    )

    submitted = captured["order_data"]
    assert submitted["type"] == "limit"
    assert submitted.get("limit_price") == pytest.approx(101.0)
