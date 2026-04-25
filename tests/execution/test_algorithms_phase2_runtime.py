from __future__ import annotations

from typing import Any

from ai_trading.core.enums import OrderSide, OrderType
from ai_trading.execution import algorithms


class _OrderManager:
    def __init__(self, *, accept: bool = True, raise_on_submit: bool = False) -> None:
        self.accept = accept
        self.raise_on_submit = raise_on_submit
        self.orders: list[Any] = []

    def submit_order(self, order: Any) -> bool:
        if self.raise_on_submit:
            raise RuntimeError("submit failed")
        self.orders.append(order)
        return self.accept


def test_vwap_executor_slices_to_minimum_size_and_preserves_metadata(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(algorithms.time, "sleep", lambda seconds: sleeps.append(seconds))
    manager = _OrderManager()
    executor = algorithms.VWAPExecutor(manager)  # type: ignore[arg-type]

    order_ids = executor.execute_vwap_order(
        "AAPL",
        OrderSide.BUY,
        total_quantity=250,
        duration_minutes=40,
        parent_order_id="parent-1",
        strategy_id="strategy-1",
    )

    assert len(order_ids) == 3
    assert [order.quantity for order in manager.orders] == [100, 100, 50]
    assert {order.order_type for order in manager.orders} == {OrderType.LIMIT}
    assert {order.execution_algorithm for order in manager.orders} == {"vwap"}
    assert {order.parent_order_id for order in manager.orders} == {"parent-1"}
    assert sleeps == [300, 300, 300]


def test_vwap_executor_returns_empty_when_submit_fails(monkeypatch) -> None:
    monkeypatch.setattr(algorithms.time, "sleep", lambda _seconds: None)
    manager = _OrderManager(accept=False)
    executor = algorithms.VWAPExecutor(manager)  # type: ignore[arg-type]

    assert executor.execute_vwap_order("MSFT", OrderSide.SELL, 100, duration_minutes=5) == []
    assert len(manager.orders) == 1


def test_twap_executor_evenly_slices_and_handles_submit_errors(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(algorithms.time, "sleep", lambda seconds: sleeps.append(seconds))
    manager = _OrderManager()
    executor = algorithms.TWAPExecutor(manager)  # type: ignore[arg-type]

    order_ids = executor.execute_twap_order("AAPL", OrderSide.BUY, 250, duration_minutes=8)

    assert len(order_ids) == 3
    assert [order.quantity for order in manager.orders] == [100, 100, 50]
    assert {order.execution_algorithm for order in manager.orders} == {"twap"}
    assert sleeps == [60, 60, 60]

    failing = algorithms.TWAPExecutor(_OrderManager(raise_on_submit=True))  # type: ignore[arg-type]
    assert failing.execute_twap_order("AAPL", OrderSide.BUY, 250, duration_minutes=8) == []


def test_implementation_shortfall_schedule_waits_and_order_types(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(algorithms.time, "sleep", lambda seconds: sleeps.append(seconds))
    manager = _OrderManager()
    executor = algorithms.ImplementationShortfall(manager)  # type: ignore[arg-type]

    order_ids = executor.execute_is_order(
        "AAPL",
        OrderSide.BUY,
        total_quantity=250,
        benchmark_price=100.0,
        urgency=0.9,
        parent_order_id="parent-2",
    )

    assert len(order_ids) == 2
    assert [order.quantity for order in manager.orders] == [158, 92]
    assert {order.order_type for order in manager.orders} == {OrderType.MARKET}
    assert {order.execution_algorithm for order in manager.orders} == {
        "implementation_shortfall"
    }
    assert sleeps == [30, 30]
    assert executor._calculate_wait_time(0.0) == 180
    assert executor._calculate_wait_time(1.0) == 30


def test_implementation_shortfall_fallback_schedule_and_submit_error(monkeypatch) -> None:
    monkeypatch.setattr(algorithms.time, "sleep", lambda _seconds: None)
    manager = _OrderManager(raise_on_submit=True)
    executor = algorithms.ImplementationShortfall(manager)  # type: ignore[arg-type]

    assert executor._calculate_execution_schedule("bad", 0.1) == [("bad", 0.5)]  # type: ignore[arg-type]
    assert executor.execute_is_order(
        "AAPL",
        OrderSide.BUY,
        total_quantity=100,
        benchmark_price=100.0,
    ) == []
