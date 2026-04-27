from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ai_trading.monitoring import alerts


def test_alert_acknowledge_resolve_and_dict_payload() -> None:
    alert = alerts.Alert(
        alerts.AlertType.EXECUTION,
        alerts.AlertSeverity.WARNING,
        "Order rejected",
        symbol="SPY",
        strategy_id="mean-reversion",
        value=4,
        threshold=3,
        metadata={"reason": "min_qty"},
    )

    alert.acknowledge("operator")
    alert.resolve()
    payload = alert.to_dict()

    assert payload["type"] == "execution"
    assert payload["severity"] == "warning"
    assert payload["acknowledged"] is True
    assert payload["resolved"] is True
    assert payload["acknowledged_by"] == "operator"
    assert payload["metadata"] == {"reason": "min_qty"}


def test_alert_manager_notifies_handlers_filters_and_updates_state() -> None:
    manager = alerts.AlertManager()
    seen: list[str] = []
    manager.add_alert_handler(lambda alert: seen.append(alert.message))

    warning = manager.create_alert(
        alerts.AlertType.SYSTEM,
        alerts.AlertSeverity.WARNING,
        "OMS stale",
    )
    critical = manager.create_alert(
        alerts.AlertType.PROVIDER_OUTAGE,
        alerts.AlertSeverity.CRITICAL,
        "Provider down",
    )

    assert seen == ["OMS stale", "Provider down"]
    assert manager.get_active_alerts(alerts.AlertSeverity.WARNING) == [warning]
    assert manager.get_alerts_by_type(alerts.AlertType.PROVIDER_OUTAGE) == [critical]
    assert manager.acknowledge_alert(warning.id, "ops") is True
    assert warning.acknowledged_by == "ops"
    assert manager.resolve_alert(warning.id) is True
    assert warning not in manager.get_active_alerts()
    assert manager.acknowledge_alert("missing") is False
    assert manager.resolve_alert("missing") is False


def test_alert_manager_caps_history_and_isolates_handler_errors() -> None:
    manager = alerts.AlertManager()
    manager.max_alerts = 2
    handled: list[str] = []

    def bad_handler(_alert: alerts.Alert) -> None:
        raise ValueError("boom")

    manager.add_alert_handler(bad_handler)
    manager.add_alert_handler(lambda alert: handled.append(alert.message))

    manager.create_alert(alerts.AlertType.SYSTEM, alerts.AlertSeverity.INFO, "one")
    manager.create_alert(alerts.AlertType.SYSTEM, alerts.AlertSeverity.INFO, "two")
    manager.create_alert(alerts.AlertType.SYSTEM, alerts.AlertSeverity.INFO, "three")

    assert [alert.message for alert in manager.alerts] == ["two", "three"]
    assert handled == ["one", "two", "three"]


def test_alert_manager_handles_creation_and_state_update_errors(monkeypatch) -> None:
    manager = alerts.AlertManager()
    original_alert = alerts.Alert
    calls = 0

    def flaky_alert(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TypeError("bad alert")
        return original_alert(*args, **kwargs)

    monkeypatch.setattr(alerts, "Alert", flaky_alert)

    fallback = manager.create_alert(alerts.AlertType.SYSTEM, alerts.AlertSeverity.INFO, "broken")

    assert fallback.alert_type == alerts.AlertType.SYSTEM
    assert fallback.severity == alerts.AlertSeverity.CRITICAL
    assert "Alert creation failed" in fallback.message

    class BrokenLock:
        def __enter__(self):
            raise TypeError("lock broken")

        def __exit__(self, *_args) -> None:
            return None

    manager._lock = BrokenLock()

    assert manager.acknowledge_alert("missing") is False
    assert manager.resolve_alert("missing") is False


def test_alert_cleanup_removes_old_alerts_in_single_iteration(monkeypatch) -> None:
    manager = alerts.AlertManager()
    manager.alert_retention_hours = 1
    old = manager.create_alert(alerts.AlertType.SYSTEM, alerts.AlertSeverity.INFO, "old")
    fresh = manager.create_alert(alerts.AlertType.SYSTEM, alerts.AlertSeverity.INFO, "fresh")
    old.timestamp = datetime.now(UTC) - timedelta(hours=2)
    fresh.timestamp = datetime.now(UTC)
    manager._cleanup_running = True

    def stop_after_sleep(_seconds: float) -> None:
        manager._cleanup_running = False

    monkeypatch.setattr(alerts, "psleep", stop_after_sleep)

    manager._cleanup_old_alerts()

    assert manager.alerts == [fresh]


def test_alert_cleanup_start_stop_and_error_path(monkeypatch) -> None:
    manager = alerts.AlertManager()
    starts: list[bool] = []
    joins: list[float | None] = []
    sleeps: list[int] = []

    class FakeThread:
        def __init__(self, *, target, daemon: bool) -> None:
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            starts.append(self.daemon)

        def join(self, timeout: float | None = None) -> None:
            joins.append(timeout)

    class BrokenLock:
        def __enter__(self):
            raise TypeError("cleanup lock broken")

        def __exit__(self, *_args) -> None:
            return None

    def stop_after_error_sleep(seconds: int) -> None:
        sleeps.append(seconds)
        manager._cleanup_running = False

    monkeypatch.setattr(alerts.threading, "Thread", FakeThread)
    manager.start_cleanup()
    manager.start_cleanup()
    manager.stop_cleanup()

    assert starts == [True]
    assert joins == [5]

    manager._lock = BrokenLock()
    manager._cleanup_running = True
    monkeypatch.setattr(alerts, "psleep", stop_after_error_sleep)

    manager._cleanup_old_alerts()

    assert sleeps == [300]


def test_risk_alert_engine_emits_threshold_breaches_and_honors_cooldown() -> None:
    manager = alerts.AlertManager()
    engine = alerts.RiskAlertEngine(manager)

    engine.check_portfolio_risk(
        {
            "max_drawdown": engine.thresholds["MAX_DRAWDOWN"] + 0.01,
            "sharpe_ratio": engine.thresholds["MIN_SHARPE_RATIO"] - 0.1,
            "var_95": engine.thresholds["MAX_VAR_95"] + 0.01,
        }
    )
    engine.check_position_risk(
        "SPY",
        {
            "position_percentage": engine.risk_params["MAX_POSITION_SIZE"] + 0.01,
            "unrealized_pnl_pct": -0.2,
        },
    )
    engine.check_execution_risk(
        {
            "fill_rate": 0.5,
            "average_slippage_bps": engine.risk_params.get("MAX_SLIPPAGE_BPS", 20) + 5,
        }
    )
    first_count = len(manager.alerts)
    engine.check_execution_risk({"fill_rate": 0.5})

    assert first_count == 7
    assert len(manager.alerts) == first_count
    assert {alert.alert_type for alert in manager.alerts} == {alerts.AlertType.RISK_LIMIT}


def test_risk_alert_engine_logs_bad_metric_payloads_and_create_failures() -> None:
    manager = alerts.AlertManager()
    engine = alerts.RiskAlertEngine(manager)

    engine.check_portfolio_risk({"max_drawdown": "bad"})
    engine.check_position_risk("SPY", {"position_percentage": "bad"})
    engine.check_execution_risk({"fill_rate": "bad"})

    class FailingManager:
        def create_alert(self, *_args, **_kwargs) -> None:
            raise ValueError("alert backend down")

    failing_engine = alerts.RiskAlertEngine(FailingManager())
    failing_engine._create_risk_alert("key", alerts.AlertSeverity.WARNING, "message")


def test_emit_runtime_alert_sets_state_and_thresholds(monkeypatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_emit(
        event: str,
        *,
        severity: str = "warning",
        details: dict[str, Any] | None = None,
        state: object | None = None,
        halt_reason: str | None = None,
    ) -> None:
        events.append((event, {"severity": severity, **(details or {})}))
        if state is not None and halt_reason:
            setattr(state, "halt_reason", halt_reason)

    state = type("State", (), {})()
    monkeypatch.setattr(alerts, "emit_runtime_alert", fake_emit)

    alerts.evaluate_runtime_alert_thresholds(
        breaker_open="broker",
        repeated_rejects=3,
        reject_threshold=3,
        data_stale_seconds=125.0,
        data_stale_threshold=120.0,
        recon_mismatch=True,
        halt_reason="AUTH_BAD_KEY",
        state=state,
    )

    assert [event for event, _payload in events] == [
        "ALERT_BREAKER_OPEN",
        "ALERT_REPEATED_REJECTS",
        "ALERT_DATA_STALE",
        "ALERT_RECON_MISMATCH",
        "ALERT_HALT_REASON",
        "ALERT_AUTH_HALT",
    ]
    assert state.halt_reason == "AUTH_BAD_KEY"


def test_emit_runtime_alert_logs_by_severity_and_ignores_unsettable_state(monkeypatch) -> None:
    logged: list[tuple[str, str, dict[str, Any]]] = []

    class Log:
        def error(self, event: str, *, extra: dict[str, Any]) -> None:
            logged.append(("error", event, extra))

        def warning(self, event: str, *, extra: dict[str, Any]) -> None:
            logged.append(("warning", event, extra))

        def info(self, event: str, *, extra: dict[str, Any]) -> None:
            logged.append(("info", event, extra))

    monkeypatch.setattr(alerts, "logger", Log())

    alerts.emit_runtime_alert("ALERT_INFO", severity="info", details={"x": 1})
    alerts.emit_runtime_alert("ALERT_WARN", severity="warning", state=1, halt_reason="NOPE")
    alerts.emit_runtime_alert("ALERT_CRIT", severity="critical", details={"x": 2})

    assert logged == [
        ("info", "ALERT_INFO", {"x": 1, "severity": "info"}),
        ("warning", "ALERT_WARN", {"severity": "warning"}),
        ("error", "ALERT_CRIT", {"x": 2, "severity": "critical"}),
    ]
