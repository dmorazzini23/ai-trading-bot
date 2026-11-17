"""Tests for allocation delta sizing and execution logging helpers."""

from types import SimpleNamespace
from typing import Any, cast

import logging

from ai_trading.core import bot_engine
from ai_trading.telemetry import runtime_state


def test_record_broker_sync_metrics_updates_state(caplog) -> None:
    """Broker sync helper should update metrics and emit structured log."""

    state = bot_engine.BotState()
    state.execution_metrics = bot_engine.ExecutionCycleMetrics()
    snapshot = SimpleNamespace(open_orders=(1, 2, 3), positions=("AAPL",))
    runtime_state.reset_broker_status()

    caplog.set_level(logging.INFO)
    bot_engine._record_broker_sync_metrics(state, cast(Any, snapshot))

    assert state.execution_metrics.open_orders == 3
    assert state.execution_metrics.positions == 1
    record = next(rec for rec in caplog.records if rec.msg == "BROKER_SYNC")
    assert record.open_orders == 3
    assert record.positions == 1
    broker_state = runtime_state.observe_broker_status()
    assert broker_state.get("connected") is True
    assert broker_state.get("status") == "connected"


def test_record_broker_sync_metrics_reports_cached_snapshot_degraded() -> None:
    state = bot_engine.BotState()
    state.execution_metrics = bot_engine.ExecutionCycleMetrics()
    snapshot = SimpleNamespace(
        open_orders=(1, 2),
        positions=("AAPL",),
        fresh=False,
        open_orders_fresh=False,
        positions_fresh=True,
        last_error="http_500",
        failed_components=("open_orders",),
        consecutive_failures=2,
        stale_age_s=30.0,
    )
    runtime_state.reset_broker_status()

    bot_engine._record_broker_sync_metrics(state, cast(Any, snapshot))

    broker_state = runtime_state.observe_broker_status()
    assert broker_state["connected"] is False
    assert broker_state["status"] == "degraded"
    assert broker_state["open_orders_count"] == 2
    assert broker_state["positions_count"] == 1
    assert broker_state["failed_components"] == ["open_orders"]
