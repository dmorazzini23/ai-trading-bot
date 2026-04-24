from __future__ import annotations

import json
import time
from datetime import timedelta
from collections.abc import Generator
from pathlib import Path

import pytest

from ai_trading.monitoring import order_health_monitor as ohm
from ai_trading.utils.time import safe_utcnow


@pytest.fixture(autouse=True)
def _clear_order_monitor_state() -> Generator[None, None, None]:
    with ohm._order_tracking_lock:
        ohm._active_orders.clear()
    ohm._order_health_monitor = None
    yield
    with ohm._order_tracking_lock:
        ohm._active_orders.clear()
    ohm._order_health_monitor = None


class _ExecutionEngine:
    def __init__(self) -> None:
        self.canceled: list[str] = []

    def _cancel_stale_order(self, order_id: str) -> bool:
        self.canceled.append(order_id)
        return True


def test_order_info_quantity_alias() -> None:
    order = ohm.OrderInfo(
        order_id="order-1",
        symbol="AAPL",
        side="buy",
        qty=17,
        submitted_time=time.time(),
        last_status="NEW",
    )

    assert order.quantity == 17


def test_check_stale_orders_invokes_execution_engine() -> None:
    engine = _ExecutionEngine()
    monitor = ohm.OrderHealthMonitor(engine)
    monitor.order_timeout_seconds = 1
    with ohm._order_tracking_lock:
        ohm._active_orders["stale"] = ohm.OrderInfo(
            order_id="stale",
            symbol="MSFT",
            side="buy",
            qty=10,
            submitted_time=time.time() - 30,
            last_status="NEW",
        )
        ohm._active_orders["fresh"] = ohm.OrderInfo(
            order_id="fresh",
            symbol="AAPL",
            side="sell",
            qty=5,
            submitted_time=time.time(),
            last_status="NEW",
        )

    monitor._check_stale_orders()

    assert engine.canceled == ["stale"]


def test_partial_fill_tracking_retries_then_abandons() -> None:
    monitor = ohm.OrderHealthMonitor(_ExecutionEngine())
    monitor.max_retry_attempts = 1
    stale_update = safe_utcnow() - timedelta(minutes=10)
    partial = ohm.PartialFillInfo(
        order_id="partial-1",
        symbol="TSLA",
        total_qty=100,
        filled_qty=25,
        remaining_qty=75,
        fill_rate=0.25,
        first_fill_time=stale_update,
        last_update=stale_update,
    )
    monitor._partial_fills[partial.order_id] = partial

    monitor._update_partial_fill_tracking()
    assert monitor._partial_fills["partial-1"].retry_count == 1

    monitor._partial_fills["partial-1"].last_update = stale_update
    monitor._update_partial_fill_tracking()
    assert "partial-1" not in monitor._partial_fills


def test_record_order_fill_updates_metrics_and_partial_state() -> None:
    monitor = ohm.OrderHealthMonitor()

    monitor.record_order_fill("order-1", fill_time=12.5, is_partial=True, fill_qty=20, total_qty=100)
    monitor.record_order_fill("order-1", fill_time=7.5, is_partial=True, fill_qty=60, total_qty=100)
    metrics = monitor._calculate_health_metrics()

    assert list(monitor._fill_times) == [12.5, 7.5]
    assert monitor._partial_fills["order-1"].remaining_qty == 40
    assert metrics.partial_fills == 1
    assert metrics.avg_fill_time == 10.0
    assert metrics.avg_fill_rate == 0.6


def test_health_summary_includes_recent_metric_trends() -> None:
    monitor = ohm.OrderHealthMonitor()
    monitor._order_metrics.append(ohm.OrderHealthMetrics(success_rate=0.8, avg_fill_time=10.0))
    monitor._order_metrics.append(
        ohm.OrderHealthMetrics(success_rate=0.9, avg_fill_time=12.5, stuck_orders=1)
    )

    summary = monitor.get_health_summary()

    assert summary["trends"] == {
        "success_rate_trend": pytest.approx(0.1),
        "fill_time_trend": 2.5,
        "stuck_orders_trend": 1,
    }
    assert summary["monitoring_active"] is False


def test_export_metrics_writes_serialized_history(tmp_path: Path) -> None:
    monitor = ohm.OrderHealthMonitor()
    monitor._order_metrics.append(
        ohm.OrderHealthMetrics(
            total_orders=3,
            success_rate=0.75,
            avg_fill_time=42.0,
            stuck_orders=1,
            partial_fills=2,
            avg_fill_rate=0.5,
        )
    )
    output = tmp_path / "order-health.json"

    monitor.export_metrics(str(output))

    exported = json.loads(output.read_text())
    assert exported == [
        {
            "timestamp": exported[0]["timestamp"],
            "total_orders": 3,
            "success_rate": 0.75,
            "avg_fill_time": 42.0,
            "stuck_orders": 1,
            "partial_fills": 2,
            "avg_fill_rate": 0.5,
        }
    ]


def test_global_monitor_helpers_reuse_singleton() -> None:
    first = ohm.get_order_health_monitor()
    second = ohm.get_order_health_monitor(_ExecutionEngine())

    assert first is second
    assert ohm.get_order_health_summary()["current_metrics"]["total_orders"] == 0
    ohm.stop_order_monitoring()
