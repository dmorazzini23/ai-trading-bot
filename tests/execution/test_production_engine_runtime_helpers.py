from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core.enums import OrderSide, OrderType, RiskLevel
from ai_trading.core.runtime_contract import UnknownExecutionModeError
from ai_trading.execution.classes import ExecutionResult, OrderRequest
from ai_trading.execution.engine import ExecutionAlgorithm, Order, OrderStatus
from ai_trading.execution import production_engine as pe


def _coordinator(monkeypatch) -> pe.ProductionExecutionCoordinator:
    coordinator = pe.ProductionExecutionCoordinator(account_equity=100_000.0)
    monkeypatch.setattr(
        coordinator.alert_manager,
        "send_trading_alert",
        lambda *_args, **_kwargs: "alert-id",
    )
    monkeypatch.setattr(
        coordinator.alert_manager,
        "send_performance_alert",
        lambda *_args, **_kwargs: "perf-alert-id",
    )
    return coordinator


def _run(coro):
    return asyncio.run(coro)


def _order(
    *,
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    quantity: int = 10,
    price: float | None = 100.0,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=cast(Any, price),
    )


def _attach_market_evidence(order: Order) -> Order:
    setattr(order, "market_data", {"atr": 2.0, "volume": 1_000_000})
    setattr(order, "historical_data", {"returns": [0.01, -0.005] * 5})
    return order


def test_run_helper_returns_coordinator() -> None:
    coordinator = _run(pe.run(account_equity=12_345.0, risk_level=RiskLevel.CONSERVATIVE))

    assert isinstance(coordinator, pe.ProductionExecutionCoordinator)
    assert coordinator.account_equity == 12_345.0
    assert coordinator.risk_level is RiskLevel.CONSERVATIVE


def test_unknown_execution_mode_fails_fast_outside_tests(monkeypatch) -> None:
    monkeypatch.setattr("ai_trading.core.runtime_contract.is_testing_mode", lambda: False)

    with pytest.raises(UnknownExecutionModeError, match="EXECUTION_MODE must be one of"):
        pe.ProductionExecutionCoordinator(account_equity=100_000.0, execution_mode="mystery")


def test_submit_order_request_rejects_invalid_request(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    request = OrderRequest(symbol="", side=OrderSide.BUY, quantity=0)

    result = _run(coordinator.submit_order_request(request))

    assert result.status == "rejected"
    assert result.order_id == request.client_order_id
    assert "Invalid order request" in result.message


def test_submit_order_request_forwards_metadata(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    captured: dict[str, Any] = {}

    async def _submit_order(**kwargs):
        captured.update(kwargs)
        return ExecutionResult(
            status="success",
            order_id="oid",
            symbol=kwargs["symbol"],
            side=kwargs["side"].value,
            quantity=kwargs["quantity"],
        )

    monkeypatch.setattr(coordinator, "submit_order", _submit_order)
    request = OrderRequest(
        symbol="msft",
        side=OrderSide.SELL,
        quantity=7,
        order_type=OrderType.LIMIT,
        price=123.45,
        strategy="unit",
        notes="note",
    )

    result = _run(coordinator.submit_order_request(request))

    assert result.status == "success"
    assert captured["symbol"] == "MSFT"
    assert captured["side"] is OrderSide.SELL
    assert captured["metadata"]["client_order_id"] == request.client_order_id
    assert captured["metadata"]["notes"] == "note"


def test_submit_order_applies_limit_preference_and_finalizes_stats(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    monkeypatch.setattr(
        pe,
        "get_settings",
        lambda: SimpleNamespace(exec_prefer_limit=True, exec_max_participation_rate=0.2),
    )
    captured: dict[str, Any] = {}

    async def _safety(order):
        captured["safety_order_type"] = order.order_type
        captured["max_participation_rate"] = order.max_participation_rate
        return {"approved": True, "reason": "ok"}

    async def _sizing(order):
        return {"final_quantity": 3}

    async def _impact(order):
        return {"estimated_slippage_bps": 1, "impact_level": "low"}

    async def _execute(order, _impact_analysis):
        captured["executed_order"] = order
        return ExecutionResult(
            status="success",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            fill_price=float(order.price),
            actual_slippage_bps=1.0,
        )

    async def _post(order, result, original_quantity):
        captured["post"] = (order.quantity, result.status, original_quantity)

    monkeypatch.setattr(coordinator, "_comprehensive_safety_check", _safety)
    monkeypatch.setattr(coordinator, "_optimize_order_size", _sizing)
    monkeypatch.setattr(coordinator, "_analyze_market_impact", _impact)
    monkeypatch.setattr(coordinator, "_execute_order_with_monitoring", _execute)
    monkeypatch.setattr(coordinator, "_post_execution_processing", _post)

    result = _run(
        coordinator.submit_order(
            "AAPL",
            OrderSide.BUY,
            10,
            order_type=OrderType.MARKET,
            price=None,
            metadata={"target_price": 101.5, "urgency_level": 0.2},
        )
    )

    assert result.status == "success"
    assert captured["safety_order_type"] is OrderType.LIMIT
    assert captured["max_participation_rate"] == pytest.approx(0.2)
    executed = cast(Any, captured["executed_order"])
    assert executed.quantity == 3
    assert float(executed.price) == pytest.approx(101.5)
    assert captured["post"] == (3, "success", 10)
    assert coordinator.execution_stats["total_orders"] == 1
    assert coordinator.execution_stats["successful_orders"] == 1


def test_submit_order_rejects_failed_safety_and_zero_sizing(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)

    async def _reject_safety(_order):
        return {"approved": False, "reason": "blocked"}

    monkeypatch.setattr(coordinator, "_comprehensive_safety_check", _reject_safety)
    rejected = _run(coordinator.submit_order("AAPL", OrderSide.BUY, 10))

    assert rejected.status == "rejected"
    assert "blocked" in rejected.message
    assert coordinator.execution_stats["rejected_orders"] == 1

    coordinator = _coordinator(monkeypatch)

    async def _approve(_order):
        return {"approved": True, "reason": "ok"}

    async def _zero(_order):
        return {"final_quantity": 0}

    monkeypatch.setattr(coordinator, "_comprehensive_safety_check", _approve)
    monkeypatch.setattr(coordinator, "_optimize_order_size", _zero)
    zero = _run(coordinator.submit_order("MSFT", OrderSide.BUY, 10))

    assert zero.status == "rejected"
    assert zero.message == "Invalid position size"
    assert coordinator.execution_stats["rejected_orders"] == 1


def test_position_tracking_resets_average_price_on_sign_flip(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    coordinator.current_positions["AAPL"] = {
        "quantity": 5,
        "avg_price": 100.0,
        "last_updated": datetime.now(UTC),
    }
    order = _order(side=OrderSide.SELL, quantity=8, price=90.0)
    order.average_fill_price = 90.0

    coordinator._update_position_tracking(order)

    assert coordinator.current_positions["AAPL"]["quantity"] == -3
    assert coordinator.current_positions["AAPL"]["avg_price"] == pytest.approx(90.0)


def test_comprehensive_safety_check_branches(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    order = _order(quantity=10)

    monkeypatch.setattr(
        coordinator.halt_manager,
        "is_trading_allowed",
        lambda: {"trading_allowed": False, "reasons": ["manual"]},
    )
    halted = _run(coordinator._comprehensive_safety_check(order))
    assert halted == {"approved": False, "reason": "Trading halted: manual"}

    monkeypatch.setattr(
        coordinator.halt_manager,
        "is_trading_allowed",
        lambda: {"trading_allowed": True, "reasons": []},
    )
    invalid_qty = _run(coordinator._comprehensive_safety_check(_order(quantity=0)))
    assert invalid_qty["reason"] == "Invalid quantity"

    monkeypatch.setattr(
        coordinator.risk_manager,
        "assess_trade_risk",
        lambda *_args: {"approved": False, "warnings": ["risk"]},
    )
    risk = _run(coordinator._comprehensive_safety_check(order))
    assert risk["reason"] == "Risk assessment failed: risk"

    monkeypatch.setattr(
        coordinator.risk_manager,
        "assess_trade_risk",
        lambda *_args: {"approved": True, "warnings": []},
    )
    coordinator.pending_orders["dup"] = _order(quantity=10)
    duplicate = _run(coordinator._comprehensive_safety_check(order))
    assert duplicate["reason"] == "Similar order recently submitted"


def test_comprehensive_safety_check_rejects_missing_real_price(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)

    result = _run(coordinator._comprehensive_safety_check(_order(price=None)))

    assert result == {
        "approved": False,
        "reason": "Real order price required for risk assessment",
    }


def test_optimize_order_size_applies_position_multiplier(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    monkeypatch.setattr(
        coordinator.position_sizer,
        "calculate_optimal_position",
        lambda *_args: {"recommended_size": 8, "warnings": ["cap"]},
    )
    monkeypatch.setattr(
        coordinator.halt_manager,
        "is_trading_allowed",
        lambda: {"trading_allowed": True, "position_size_multiplier": 0.5},
    )

    result = _run(coordinator._optimize_order_size(_attach_market_evidence(_order(quantity=10))))

    assert result["original_quantity"] == 10
    assert result["recommended_quantity"] == 8
    assert result["final_quantity"] == 5
    assert result["sizing_warnings"] == ["cap"]


def test_optimize_order_size_blocks_weak_evidence_warning(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    monkeypatch.setattr(
        coordinator.position_sizer,
        "calculate_optimal_position",
        lambda *_args: {"recommended_size": 8, "warnings": ["weak evidence sizing"]},
    )
    monkeypatch.setattr(
        coordinator.halt_manager,
        "is_trading_allowed",
        lambda: {"trading_allowed": True, "position_size_multiplier": 1.0},
    )

    result = _run(coordinator._optimize_order_size(_attach_market_evidence(_order(quantity=10))))

    assert result["final_quantity"] == 0
    assert result["sizing_warnings"] == ["weak evidence sizing"]


def test_optimize_order_size_rejects_without_real_market_evidence(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)

    no_atr = _run(coordinator._optimize_order_size(_order(quantity=10)))
    no_history_order = _order(quantity=10)
    setattr(no_history_order, "market_data", {"atr": 2.0})
    no_history = _run(coordinator._optimize_order_size(no_history_order))

    assert no_atr["final_quantity"] == 0
    assert no_atr["sizing_warnings"] == ["Real ATR market data required for position sizing"]
    assert no_history["final_quantity"] == 0
    assert no_history["sizing_warnings"] == ["Real return history required for position sizing"]


def test_market_impact_and_execution_monitoring(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)

    high = _run(coordinator._analyze_market_impact(_order(quantity=20_000, price=100.0)))
    medium = _run(coordinator._analyze_market_impact(_order(quantity=2_000, price=100.0)))
    low = _run(coordinator._analyze_market_impact(_order(quantity=10, price=100.0)))

    assert high["impact_level"] == "high"
    assert high["recommended_algorithm"] is ExecutionAlgorithm.TWAP
    assert medium["recommended_algorithm"] is ExecutionAlgorithm.VWAP
    assert low["recommended_algorithm"] is ExecutionAlgorithm.MARKET

    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(pe.asyncio, "sleep", _sleep)
    coordinator.execution_mode = "sim"
    order = _order(quantity=10, price=100.0)
    result = _run(coordinator._execute_order_with_monitoring(order, {"estimated_slippage_bps": 5}))

    assert result.status == "success"
    assert order.status is OrderStatus.FILLED
    assert order.id in coordinator.completed_orders
    assert order.id not in coordinator.pending_orders
    assert coordinator.current_positions["AAPL"]["quantity"] == 10
    assert result.actual_slippage_bps == pytest.approx(5.0)


def test_live_execution_requires_canonical_oms_pretrade(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    coordinator.execution_mode = "live"
    order = _order(quantity=10, price=100.0)

    result = _run(coordinator._execute_order_with_monitoring(order, {"estimated_slippage_bps": 5}))

    assert result.status == "failed"
    assert result.error_code == "canonical_oms_pretrade_required"
    assert order.status is OrderStatus.REJECTED


def test_live_execution_does_not_use_broker_adapter(monkeypatch) -> None:
    class Adapter:
        provider = "alpaca"

        def __init__(self) -> None:
            self.payload: dict[str, Any] | None = None

        def submit_order(self, payload):
            self.payload = dict(payload)
            return {"id": "broker-1", "status": "accepted", "client_order_id": payload["client_order_id"]}

    adapter = Adapter()
    coordinator = _coordinator(monkeypatch)
    coordinator.execution_mode = "live"
    coordinator.broker_adapter = adapter
    order = _order(quantity=10, price=100.0)

    result = _run(coordinator._execute_order_with_monitoring(order, {"estimated_slippage_bps": 5}))

    assert result.status == "failed"
    assert result.error_code == "canonical_oms_pretrade_required"
    assert adapter.payload is None
    assert order.status is OrderStatus.REJECTED


def test_live_execution_suppresses_adapter_partial_fill_simulation(monkeypatch) -> None:
    class Adapter:
        provider = "alpaca"

        def submit_order(self, payload):
            return {
                "id": "broker-partial-1",
                "status": "partially_filled",
                "client_order_id": payload["client_order_id"],
                "filled_qty": "2.5",
                "filled_avg_price": "101.25",
            }

    coordinator = _coordinator(monkeypatch)
    coordinator.execution_mode = "live"
    coordinator.broker_adapter = Adapter()
    order = _order(quantity=10, price=100.0)

    result = _run(coordinator._execute_order_with_monitoring(order, {"estimated_slippage_bps": 5}))

    assert result.status == "failed"
    assert result.error_code == "canonical_oms_pretrade_required"
    assert order.status is OrderStatus.REJECTED
    assert "broker-partial-1" not in coordinator.pending_orders


def test_post_execution_processing_alert_paths(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    trading_alerts: list[tuple[object, ...]] = []
    performance_alerts: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        coordinator.alert_manager,
        "send_trading_alert",
        lambda *args: trading_alerts.append(args),
    )
    monkeypatch.setattr(
        coordinator.alert_manager,
        "send_performance_alert",
        lambda *args: performance_alerts.append(args),
    )
    order = _order(quantity=1_000, price=100.0)
    success = ExecutionResult(
        status="success",
        order_id=order.id,
        symbol=order.symbol,
        side=order.side.value,
        quantity=order.quantity,
        fill_price=100.0,
        actual_slippage_bps=999.0,
    )

    _run(coordinator._post_execution_processing(order, success, original_quantity=1_000))

    assert trading_alerts
    assert performance_alerts

    failed = ExecutionResult(
        status="failed",
        order_id="bad",
        symbol="AAPL",
        quantity=5,
        message="no fill",
    )
    _run(coordinator._post_execution_processing(_order(quantity=5), failed, original_quantity=5))

    assert any(args[0] == "Order Execution Failed" for args in trading_alerts)


def test_position_tracking_statistics_summary_and_cancel(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    buy = _order(quantity=10, price=100.0)
    buy.average_fill_price = cast(Any, 100.0)
    sell = _order(side=OrderSide.SELL, quantity=4, price=110.0)
    sell.average_fill_price = cast(Any, 110.0)

    coordinator._update_position_tracking(buy)
    coordinator._update_position_tracking(sell)

    assert coordinator.current_positions["AAPL"]["quantity"] == 6

    _run(
        coordinator._update_execution_statistics(
            {"status": "success", "actual_slippage_bps": 4.0},
            execution_time_ms=10.0,
        )
    )
    _run(coordinator._update_execution_statistics({"status": "rejected"}, execution_time_ms=5.0))
    _run(coordinator._update_execution_statistics({"status": "accepted"}, execution_time_ms=5.0))

    assert coordinator.execution_stats["total_orders"] == 3
    assert coordinator.execution_stats["successful_orders"] == 1
    assert coordinator.execution_stats["rejected_orders"] == 1
    summary = coordinator.get_execution_summary()
    assert summary["success_rate_pct"] == pytest.approx(100.0 / 3.0)
    assert summary["average_slippage_bps"] == pytest.approx(4.0)
    assert coordinator.get_current_positions() == coordinator.current_positions

    pending = _order(symbol="MSFT", quantity=3, price=50.0)
    coordinator.pending_orders[pending.id] = pending
    pending_snapshot = coordinator.get_pending_orders()
    assert pending_snapshot[pending.id]["symbol"] == "MSFT"

    missing = _run(coordinator.cancel_order("missing"))
    canceled = _run(coordinator.cancel_order(pending.id))

    assert missing.status == "error"
    assert canceled.status == "success"
    assert pending.status is OrderStatus.CANCELED
    assert pending.id in coordinator.rejected_orders


def test_stop_snapshot_helpers_and_account_update(monkeypatch) -> None:
    coordinator = _coordinator(monkeypatch)
    coordinator.current_positions = cast(dict[str, dict[str, Any]], {
        "AAPL": {"quantity": "5"},
        "BAD": {"quantity": "not-a-number"},
        "MSFT": cast(Any, SimpleNamespace(qty=2)),
    })

    ledger_positions, source = coordinator._snapshot_positions_for_stop_checks()
    assert source == "ledger"
    assert [pos.symbol for pos in ledger_positions] == ["AAPL", "BAD", "MSFT"]
    assert [pos.qty for pos in ledger_positions] == [5.0, 0.0, 2.0]

    broker_positions = [SimpleNamespace(symbol="TSLA", qty=1)]
    setattr(cast(Any, coordinator), "broker", SimpleNamespace(list_positions=lambda: broker_positions))
    broker_snapshot, source = coordinator._snapshot_positions_for_stop_checks()
    assert broker_snapshot is broker_positions
    assert source == "broker"
    assert coordinator._resolve_broker_like_interface() is getattr(coordinator, "broker")

    updated: list[float] = []
    monkeypatch.setattr(coordinator.halt_manager, "update_equity", lambda value: updated.append(value))
    coordinator.update_account_equity(123_000.0)

    assert coordinator.account_equity == 123_000.0
    assert updated == [123_000.0]


def test_recommend_algorithm_and_recent_duplicate_detection() -> None:
    coordinator = pe.ProductionExecutionCoordinator(account_equity=10_000.0)

    assert coordinator._recommend_execution_algorithm("high") is ExecutionAlgorithm.TWAP
    assert coordinator._recommend_execution_algorithm("medium") is ExecutionAlgorithm.VWAP
    assert coordinator._recommend_execution_algorithm("low") is ExecutionAlgorithm.MARKET

    recent = _order(quantity=5)
    old = _order(quantity=5)
    old.created_at = datetime.now(UTC) - timedelta(minutes=5)
    coordinator.pending_orders = {"recent": recent}

    assert coordinator._has_recent_similar_order(_order(quantity=5)) is True
    coordinator.pending_orders = {"old": old}
    assert coordinator._has_recent_similar_order(_order(quantity=5)) is False
