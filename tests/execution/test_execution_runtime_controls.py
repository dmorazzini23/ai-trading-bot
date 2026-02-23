from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading as lt


def _engine_stub() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {}
    engine._cycle_submitted_orders = 0
    engine._cycle_order_outcomes = []
    engine._recent_order_intents = {}
    engine._pending_new_actions_this_cycle = 0
    engine.marketable_limit_slippage_bps = 10
    engine._capacity_broker = lambda client: client
    return engine


def test_pending_new_timeout_policy_cancel(monkeypatch, caplog):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    stale_order = SimpleNamespace(
        id="ord-1",
        symbol="AAPL",
        side="buy",
        qty="1",
        status="pending_new",
        created_at=now_dt - timedelta(seconds=120),
    )
    engine.trading_client = SimpleNamespace(list_orders=lambda status="open": [stale_order])
    canceled: list[str] = []
    engine._cancel_order_alpaca = lambda order_id: canceled.append(str(order_id))
    engine._replace_limit_order_with_marketable = lambda **_: None

    monkeypatch.setenv("AI_TRADING_PENDING_NEW_POLICY", "cancel")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_TIMEOUT_SEC", "30")
    monkeypatch.setenv("AI_TRADING_PENDING_NEW_MAX_ACTIONS_PER_CYCLE", "2")

    caplog.set_level(logging.WARNING)
    engine._apply_pending_new_timeout_policy()

    assert canceled == ["ord-1"]
    assert engine._pending_new_actions_this_cycle == 1
    assert any(record.message == "PENDING_NEW_TIMEOUT_ACTION" for record in caplog.records)


def test_duplicate_intent_window(monkeypatch):
    engine = _engine_stub()
    clock = {"value": 100.0}
    monkeypatch.setenv("AI_TRADING_DUPLICATE_INTENT_WINDOW_SEC", "60")
    monkeypatch.setattr(lt, "monotonic_time", lambda: clock["value"])

    engine._record_order_intent("AAPL", "buy")

    clock["value"] = 130.0
    assert engine._should_suppress_duplicate_intent("AAPL", "buy") is True

    clock["value"] = 200.0
    assert engine._should_suppress_duplicate_intent("AAPL", "buy") is False


def test_order_pacing_cap_warning_cooldown(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CAP_LOG_COOLDOWN_SEC", "300")

    assert engine._should_emit_order_pacing_cap_log(now_ts=100.0) is True
    assert engine._should_emit_order_pacing_cap_log(now_ts=200.0) is False
    assert engine._should_emit_order_pacing_cap_log(now_ts=401.0) is True


def test_order_pacing_cap_warning_cooldown_zero_disables(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_ORDER_PACING_CAP_LOG_COOLDOWN_SEC", "0")

    assert engine._should_emit_order_pacing_cap_log(now_ts=100.0) is True
    assert engine._should_emit_order_pacing_cap_log(now_ts=101.0) is True


def test_order_pacing_cap_log_level_warmup(monkeypatch):
    engine = _engine_stub()
    monkeypatch.setenv("AI_TRADING_WARMUP_MODE", "1")
    assert engine._order_pacing_cap_log_level() == "info"


def test_order_pacing_cap_log_level_runtime(monkeypatch):
    engine = _engine_stub()
    monkeypatch.delenv("AI_TRADING_WARMUP_MODE", raising=False)
    assert engine._order_pacing_cap_log_level() == "warning"


def test_execution_kpi_snapshot_and_alerts(monkeypatch, caplog):
    engine = _engine_stub()
    now_dt = datetime.now(UTC)
    engine._cycle_order_outcomes = [
        {"status": "pending_new", "duration_s": 12.0, "ack_timed_out": True},
        {"status": "canceled", "duration_s": 20.0, "ack_timed_out": False},
    ]
    engine._list_open_orders_snapshot = lambda: [
        SimpleNamespace(status="pending_new", created_at=now_dt - timedelta(seconds=180))
    ]

    monkeypatch.setenv("AI_TRADING_EXEC_KPI_ALERTS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_KPI_MIN_FILL_RATIO", "0.90")
    monkeypatch.setenv("AI_TRADING_KPI_MAX_CANCEL_RATIO", "0.10")
    monkeypatch.setenv("AI_TRADING_KPI_MAX_MEDIAN_PENDING_SEC", "5")
    monkeypatch.setenv("AI_TRADING_KPI_PENDING_AGE_ALERT_SEC", "60")

    emitted: list[str] = []
    import ai_trading.monitoring.alerts as alerts_mod

    def _emit_runtime_alert(event: str, **kwargs):
        del kwargs
        emitted.append(event)

    monkeypatch.setattr(alerts_mod, "emit_runtime_alert", _emit_runtime_alert)

    caplog.set_level(logging.INFO)
    engine._emit_cycle_execution_kpis()

    assert any(record.message == "EXECUTION_KPI_SNAPSHOT" for record in caplog.records)
    assert "ALERT_EXEC_KPI_LOW_FILL_RATIO" in emitted
    assert "ALERT_EXEC_KPI_HIGH_CANCEL_RATIO" in emitted
    assert "ALERT_EXEC_KPI_MEDIAN_PENDING_HIGH" in emitted
    assert "ALERT_EXEC_KPI_OPEN_PENDING_AGED" in emitted


def test_execution_kpi_snapshot_includes_lockout_and_skip_reasons(monkeypatch, caplog):
    engine = _engine_stub()
    engine._cycle_order_outcomes = [
        {"status": "skipped", "reason": "broker_lock", "duration_s": 0.1, "ack_timed_out": False},
        {"status": "failed", "reason": "submit_exception", "duration_s": 0.2, "ack_timed_out": False},
        {"status": "filled", "duration_s": 0.3, "ack_timed_out": False},
    ]
    engine._list_open_orders_snapshot = lambda: []
    engine._broker_lock_reason = "unauthorized"
    engine._broker_locked_until = 160.0
    monkeypatch.setattr(lt, "monotonic_time", lambda: 100.0)

    caplog.set_level(logging.INFO)
    engine._emit_cycle_execution_kpis()

    record = next(rec for rec in caplog.records if rec.message == "EXECUTION_KPI_SNAPSHOT")
    assert record.submitted == 1
    assert record.failed == 1
    assert record.skipped == 1
    assert record.skip_reason_counts == {"broker_lock": 1}
    assert record.broker_lock_active is True
    assert record.broker_lock_reason == "unauthorized"
    assert record.broker_lock_ttl_s == pytest.approx(60.0, abs=0.1)
