import sys
from decimal import Decimal
from typing import Any
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


import ai_trading.execution.live_trading as lt
from ai_trading.execution import guards


@pytest.fixture
def engine_factory(monkeypatch):
    """Provide a minimally wired execution engine for limit order tests."""

    monkeypatch.setitem(
        sys.modules,
        "flask",
        SimpleNamespace(Flask=lambda *_, **__: None, jsonify=lambda obj: obj),
    )

    monkeypatch.setattr(lt, "_safe_mode_guard", lambda *_, **__: False)
    guards.STATE.pdt = guards.PDTState()
    guards.STATE.shadow_cycle = False
    guards.STATE.shadow_cycle_forced = False

    def _capacity_stub(symbol, side, price_hint, quantity, broker, account_snapshot, preflight_fn=None):
        return lt.CapacityCheck(True, int(quantity))

    monkeypatch.setattr(lt, "_call_preflight_capacity", _capacity_stub)
    monkeypatch.setattr(lt, "get_tick_size", lambda symbol: Decimal("0.01"))

    def _build_engine(execute_behavior=None, *, use_real_submit: bool = False, trading_client=None):
        engine = object.__new__(lt.ExecutionEngine)
        engine._refresh_settings = lambda: None
        engine.is_initialized = True
        engine._ensure_initialized = lambda: True
        engine._pre_execution_checks = lambda: True
        engine._get_account_snapshot = lambda: None
        engine._should_skip_for_pdt = lambda _account, _closing: (False, None, {})
        if execute_behavior is None:
            engine._execute_with_retry = lambda submit_fn, order_data: submit_fn(order_data)
        else:
            engine._execute_with_retry = lambda submit_fn, order_data: execute_behavior(order_data)
        if use_real_submit:
            engine._submit_order_to_alpaca = lt.ExecutionEngine._submit_order_to_alpaca.__get__(engine, lt.ExecutionEngine)
        else:
            engine._submit_order_to_alpaca = lambda order_data: {"id": "abc123", "order_data": order_data}
        engine.shadow_mode = False
        engine.trading_client = trading_client if trading_client is not None else SimpleNamespace()
        engine.stats = {
            "total_execution_time": 0.0,
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
        }
        monkeypatch.setattr(
            lt,
            "get_trading_config",
            lambda: SimpleNamespace(
                nbbo_required_for_limit=False,
                execution_require_realtime_nbbo=False,
                execution_market_on_degraded=False,
                degraded_feed_mode="widen",
                degraded_feed_limit_widen_bps=0,
                min_quote_freshness_ms=1500,
            ),
        )
        monkeypatch.setattr(lt.provider_monitor, "is_disabled", lambda *_a, **_k: False)
        monkeypatch.setattr(lt, "_require_bid_ask_quotes", lambda: False)
        monkeypatch.setattr(lt, "guard_shadow_active", lambda: False)
        monkeypatch.setattr(lt, "is_safe_mode_active", lambda: False)
        monkeypatch.setattr(lt, "pdt_guard", lambda *a, **k: True)
        return engine

    return _build_engine


def test_submit_limit_order_success_path(engine_factory):
    engine = engine_factory(lambda order_data: {"id": "ok", **order_data})

    result = engine.submit_limit_order("AAPL", "buy", 10, 123.45)

    assert result["id"] == "ok"
    assert engine.stats["successful_orders"] == 1
    assert engine.stats["failed_orders"] == 0


def test_submit_limit_order_capacity_precheck_runs_once(engine_factory, monkeypatch):
    calls: list[tuple[Any, Any, Any, Any]] = []

    def _capacity_stub(symbol, side, price_hint, quantity, broker, account_snapshot, preflight_fn=None):
        calls.append((symbol, side, price_hint, quantity))
        return lt.CapacityCheck(True, int(quantity))

    monkeypatch.setattr(lt, "_call_preflight_capacity", _capacity_stub)
    engine = engine_factory(lambda order_data: {"id": "ok", **order_data})

    result = engine.submit_limit_order("AAPL", "buy", 10, 123.45)

    assert result["id"] == "ok"
    assert len(calls) == 1
    assert calls[0][2] == pytest.approx(123.45)


def test_execute_order_uses_capacity_suggested_qty(engine_factory, monkeypatch):
    captured: dict[str, Any] = {}

    def _capacity_stub(symbol, side, price_hint, quantity, broker, account_snapshot, preflight_fn=None):
        return lt.CapacityCheck(True, 3)

    def _submit_limit_stub(symbol, side, quantity, limit_price, **kwargs):
        captured["quantity"] = quantity
        captured["limit_price"] = limit_price
        return {
            "id": "ok",
            "symbol": symbol,
            "side": side,
            "status": "accepted",
            "qty": quantity,
            "filled_qty": "0",
        }

    monkeypatch.setattr(lt, "_call_preflight_capacity", _capacity_stub)
    engine = engine_factory(lambda order_data: {"id": "ok", **order_data})
    engine.submit_limit_order = _submit_limit_stub

    result = engine.execute_order("AAPL", "buy", 10, order_type="limit", limit_price=123.45)

    assert result is not None
    assert captured["quantity"] == 3
    assert result.requested_quantity == 3


def test_submit_limit_order_returns_broker_payload(engine_factory, monkeypatch):
    submissions: list[dict[str, object]] = []

    def _fake_submit(order_payload):
        submissions.append(order_payload)
        return {"id": "broker-ack", "status": "accepted", **order_payload}

    fake_client = SimpleNamespace(submit_order=_fake_submit)
    monkeypatch.setenv("PYTEST_RUNNING", "1")

    engine = engine_factory(None, use_real_submit=True, trading_client=fake_client)

    result = engine.submit_limit_order("AAPL", "buy", 5, 123.45)

    assert result["id"] == "broker-ack"
    assert result["status"] == "accepted"
    assert engine.stats["successful_orders"] == 1
    assert engine.stats["failed_orders"] == 0
    assert submissions and submissions[0]["symbol"] == "AAPL"


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


def test_execute_order_uses_limit_with_fallback(engine_factory, monkeypatch, caplog):
    monkeypatch.setenv("EXECUTION_FALLBACK_LIMIT_BUFFER_BPS", "100")
    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "widen")
    monkeypatch.setenv("TRADING__DEGRADED_FEED_LIMIT_WIDEN_BPS", "0")
    from ai_trading.config import runtime as _runtime_cfg

    _runtime_cfg.reload_trading_config()
    monkeypatch.setattr(lt.provider_monitor, "is_disabled", lambda *_a, **_k: False)
    monkeypatch.setattr(lt, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(lt, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(lt, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(lt, "pdt_guard", lambda *a, **k: True)

    captured: dict[str, Any] = {}

    def _capture(order_data: dict[str, Any]) -> dict[str, Any]:
        captured["order_data"] = dict(order_data)
        return {"id": "ok", **order_data}

    engine = engine_factory(_capture)
    caplog.set_level("WARNING", logger=lt.logger.name)

    result = engine.execute_order(
        "AAPL",
        "buy",
        5,
        price=100.0,
        annotations={"using_fallback_price": True},
    )

    assert result == "ok"
    submitted = captured["order_data"]
    assert submitted.get("type") == "limit"
    assert submitted.get("limit_price") == pytest.approx(100.0)
    messages = [record.message for record in caplog.records]
    assert "QUOTE_QUALITY_BLOCKED" not in messages


def test_execute_order_fallback_reject_no_market_retry_by_default(engine_factory, monkeypatch, caplog):
    def _reject_limit(*_args, **_kwargs):
        raise lt.NonRetryableBrokerError(
            "rejected",
            code="400",
            detail="price outside allowed band",
        )

    market_calls: list[tuple[str, str, int]] = []

    def _submit_market(symbol, side, quantity, **_kwargs):
        market_calls.append((symbol, side, quantity))
        return {"id": "mk-1", "status": "accepted", "qty": quantity}

    monkeypatch.setattr(
        lt,
        "get_trading_config",
        lambda: SimpleNamespace(
            nbbo_required_for_limit=False,
            execution_require_realtime_nbbo=False,
            execution_market_on_degraded=False,
            execution_market_on_fallback=False,
            degraded_feed_mode="widen",
            degraded_feed_limit_widen_bps=0,
            min_quote_freshness_ms=1500,
        ),
    )

    engine = engine_factory()
    engine.submit_limit_order = _reject_limit
    engine.submit_market_order = _submit_market

    caplog.set_level("WARNING", logger=lt.logger.name)
    result = engine.execute_order(
        "AAPL",
        "buy",
        5,
        order_type="limit",
        limit_price=100.0,
        annotations={"using_fallback_price": True},
    )

    assert result is None
    assert market_calls == []
    assert all(record.msg != "ORDER_DOWNGRADED_TO_MARKET" for record in caplog.records)


def test_execute_order_fallback_reject_market_retry_when_enabled(engine_factory, monkeypatch, caplog):
    def _reject_limit(*_args, **_kwargs):
        raise lt.NonRetryableBrokerError(
            "rejected",
            code="400",
            detail="price outside allowed band",
        )

    market_calls: list[tuple[str, str, int]] = []

    def _submit_market(symbol, side, quantity, **_kwargs):
        market_calls.append((symbol, side, quantity))
        return {
            "id": "mk-1",
            "status": "accepted",
            "qty": quantity,
            "filled_qty": "0",
            "symbol": symbol,
            "side": side,
        }

    engine = engine_factory()
    monkeypatch.setattr(
        lt,
        "get_trading_config",
        lambda: SimpleNamespace(
            nbbo_required_for_limit=False,
            execution_require_realtime_nbbo=False,
            execution_market_on_degraded=False,
            execution_market_on_fallback=True,
            degraded_feed_mode="widen",
            degraded_feed_limit_widen_bps=0,
            min_quote_freshness_ms=1500,
        ),
    )
    engine.submit_limit_order = _reject_limit
    engine.submit_market_order = _submit_market

    caplog.set_level("WARNING", logger=lt.logger.name)
    result = engine.execute_order(
        "AAPL",
        "buy",
        5,
        order_type="limit",
        limit_price=100.0,
        annotations={"using_fallback_price": True},
    )

    assert result is not None
    assert market_calls == [("AAPL", "buy", 5)]
    assert any(record.msg == "ORDER_DOWNGRADED_TO_MARKET" for record in caplog.records)


def test_bracket_fallback_reuses_normalized_payload(engine_factory, monkeypatch):
    class _BracketClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def submit_order(self, **kwargs):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                raise TypeError("nested bracket unsupported")
            return {"id": "ok", "status": "accepted", **kwargs}

    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    client = _BracketClient()
    engine = engine_factory(None, use_real_submit=True, trading_client=client)

    result = engine.submit_limit_order(
        "AAPL",
        "buy",
        5,
        123.45,
        order_class="bracket",
        take_profit=130.0,
        stop_loss=120.0,
    )

    assert result is not None
    assert len(client.calls) == 2
    assert "qty" in client.calls[-1]
    assert client.calls[-1]["qty"] == 5
    assert "quantity" not in client.calls[-1]
    assert "order_class" not in client.calls[-1]


def test_ttl_replacement_price_uses_tick_quantization(engine_factory, monkeypatch):
    monkeypatch.setattr(lt, "get_tick_size", lambda _symbol: Decimal("0.1"))
    engine = engine_factory()
    captured: dict[str, Any] = {}
    engine.order_ttl_seconds = 20
    engine._cancel_order_alpaca = lambda _order_id: True

    def _capture_submit(payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = dict(payload)
        return {
            "id": "replace-1",
            "status": "accepted",
            "qty": payload.get("quantity", 0),
            "symbol": payload.get("symbol"),
            "side": payload.get("side"),
            "client_order_id": payload.get("client_order_id"),
        }

    engine._submit_order_to_alpaca = _capture_submit

    replacement = engine._replace_limit_order_with_marketable(
        symbol="AAPL",
        side="buy",
        qty=2,
        existing_order_id="old-order",
        client_order_id="cid-old",
        order_data_snapshot={"symbol": "AAPL", "side": "buy", "quantity": 2, "type": "limit"},
        limit_price=100.123,
    )

    assert replacement is not None
    assert captured["payload"]["limit_price"] == pytest.approx(100.2)


def test_submit_market_order_pdt_lockout_logs(caplog):
    account_snapshot = {
        "pattern_day_trader": True,
        "daytrade_limit": 3,
        "daytrade_count": 4,
        "active": True,
        "limit": 3,
        "count": 4,
    }

    engine = object.__new__(lt.ExecutionEngine)
    engine._refresh_settings = lambda: None
    engine.is_initialized = True
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True
    engine._is_circuit_breaker_open = lambda: False
    engine._broker_lock_suppressed = lambda **_: False
    engine.shadow_mode = False
    engine.trading_client = SimpleNamespace()
    engine._cycle_account = account_snapshot
    engine._cycle_account_fetched = True
    engine.stats = {
        "capacity_skips": 0,
        "skipped_orders": 0,
        "total_orders": 0,
        "successful_orders": 0,
        "failed_orders": 0,
        "total_execution_time": 0.0,
    }
    engine._execute_with_retry = MagicMock()
    engine._submit_order_to_alpaca = MagicMock()

    caplog.set_level("DEBUG", logger=lt.logger.name)

    result = engine.submit_market_order("AAPL", "buy", 1)

    assert result is None
    engine._execute_with_retry.assert_not_called()
    messages = [record.message for record in caplog.records]
    assert any(msg.startswith("PDT_PREFLIGHT_CHECKED") for msg in messages)
    assert any(record.message == "ORDER_SKIPPED_NONRETRYABLE" for record in caplog.records)
    detail_records = [
        record for record in caplog.records if record.message.startswith("ORDER_SKIPPED_NONRETRYABLE_DETAIL")
    ]
    assert detail_records, "Expected ORDER_SKIPPED_NONRETRYABLE_DETAIL log entry"
    detail_message = detail_records[0].message
    for key in ("pattern_day_trader", "daytrade_limit", "daytrade_count", "active", "limit", "count"):
        assert key in detail_message


def test_execute_order_records_skip_outcome_for_duplicate_intent(engine_factory, caplog):
    engine = engine_factory()
    engine._cycle_order_outcomes = []
    engine._should_suppress_duplicate_intent = lambda *_args, **_kwargs: True

    caplog.set_level("INFO", logger=lt.logger.name)
    result = engine.execute_order("AAPL", "buy", 1, order_type="market")

    assert result is None
    assert engine._cycle_order_outcomes
    assert engine._cycle_order_outcomes[-1]["status"] == "skipped"
    assert engine._cycle_order_outcomes[-1]["reason"] == "duplicate_intent"
    assert any(record.msg == "ORDER_SUBMIT_SKIPPED" for record in caplog.records)


def test_execute_order_records_skip_outcome_for_cycle_duplicate_intent(engine_factory, caplog):
    engine = engine_factory()
    engine._cycle_order_outcomes = []
    engine._reserve_cycle_intent = lambda *_args, **_kwargs: False

    caplog.set_level("INFO", logger=lt.logger.name)
    result = engine.execute_order("AAPL", "buy", 1, order_type="market")

    assert result is None
    assert engine._cycle_order_outcomes
    assert engine._cycle_order_outcomes[-1]["status"] == "skipped"
    assert engine._cycle_order_outcomes[-1]["reason"] == "cycle_duplicate_intent"
    assert any(record.msg == "ORDER_SUBMIT_SKIPPED" for record in caplog.records)


def test_execute_order_records_failure_outcome_on_submit_exception(engine_factory, caplog):
    engine = engine_factory()
    engine._cycle_order_outcomes = []

    def _raise_submit_exception(*_args, **_kwargs):
        raise lt.APIError("order submit failed")

    engine.submit_market_order = _raise_submit_exception

    caplog.set_level("ERROR", logger=lt.logger.name)
    result = engine.execute_order("AAPL", "buy", 1, order_type="market")

    assert result is None
    assert engine._cycle_order_outcomes
    assert engine._cycle_order_outcomes[-1]["status"] == "failed"
    assert engine._cycle_order_outcomes[-1]["reason"] == "submit_exception"
    assert any(record.msg == "ORDER_SUBMIT_FAILED" for record in caplog.records)
