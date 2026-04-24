from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.safety import monitoring


class _FakeThread:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.joined_timeout = None

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.joined_timeout = timeout


def test_safety_monitor_start_stop_and_metric_updates(monkeypatch):
    threads: list[_FakeThread] = []

    def fake_thread(*, target, daemon=False):
        thread = _FakeThread(target, daemon=daemon)
        threads.append(thread)
        return thread

    monkeypatch.setattr(monitoring.threading, "Thread", fake_thread)
    monitor = monitoring.SafetyMonitor()
    alerts: list[dict[str, object]] = []
    monitor.add_alert_callback(alerts.append)

    monitor.start_monitoring()
    monitor.start_monitoring()
    monitor.update_metrics(total_portfolio_value=100_000.0, available_cash=50_000.0, new_order=True)
    monitor.update_metrics(unknown_metric=123)
    monitor.stop_monitoring()

    assert monitor.state is monitoring.TradingState.STOPPED
    assert not monitor.is_monitoring
    assert len(threads) == 1
    assert threads[0].started
    assert threads[0].joined_timeout == 5
    assert monitor.metrics["orders_this_minute"] == 1
    assert monitor.metrics["available_cash"] == 50_000.0
    assert "unknown_metric" not in monitor.metrics
    assert [alert["severity"] for alert in alerts] == ["info", "info"]


def test_emergency_stop_runs_actions_and_blocks_resume_until_reset():
    monitor = monitoring.SafetyMonitor()
    calls: list[str] = []

    monitor.add_emergency_action(lambda reason: calls.append(reason))
    monitor.add_emergency_action(lambda _reason: (_ for _ in ()).throw(ValueError("action failed")))

    monitor.emergency_stop("manual")

    assert monitor.emergency_stop_triggered
    assert monitor.state is monitoring.TradingState.EMERGENCY_STOP
    assert calls == ["manual"]
    assert not monitor.resume_trading()
    assert not monitor.reset_emergency_stop("wrong")
    assert monitor.reset_emergency_stop("RESET_AUTHORIZED")
    assert not monitor.emergency_stop_triggered
    assert monitor.state is monitoring.TradingState.STOPPED
    assert monitor.metrics["failed_orders_count"] == 0
    assert monitor.resume_trading("authorized")
    assert monitor.state is monitoring.TradingState.RUNNING


def test_pause_trading_noops_during_emergency_stop():
    monitor = monitoring.SafetyMonitor()
    monitor.state = monitoring.TradingState.EMERGENCY_STOP

    monitor.pause_trading("ignored")

    assert monitor.state is monitoring.TradingState.EMERGENCY_STOP


def test_check_safety_thresholds_reports_all_violation_types():
    monitor = monitoring.SafetyMonitor()
    monitor.metrics.update(
        daily_pnl=-7_500.0,
        total_portfolio_value=100_000.0,
        current_drawdown=0.20,
        available_cash=100.0,
        orders_this_minute=99,
        failed_orders_count=20,
    )

    violations = monitor.check_safety_thresholds()

    assert [violation["type"] for violation in violations] == [
        "daily_loss_limit",
        "max_drawdown",
        "low_cash",
        "order_rate_limit",
        "failed_orders",
    ]
    assert monitor.get_system_health()["violations"] == violations


def test_send_alert_handles_callback_failures():
    monitor = monitoring.SafetyMonitor()
    received: list[dict[str, object]] = []
    monitor.add_alert_callback(received.append)
    monitor.add_alert_callback(lambda _alert: (_ for _ in ()).throw(TypeError("bad callback")))

    monitor._send_alert(monitoring.AlertSeverity.CRITICAL, "critical message")

    assert received
    assert received[0]["severity"] == "critical"
    assert received[0]["message"] == "critical message"


def test_kill_switch_detects_file_and_triggers_monitor(tmp_path):
    monitor = monitoring.SafetyMonitor()
    reasons: list[str] = []
    monitor.emergency_stop = lambda reason: reasons.append(reason)
    kill_switch = monitoring.KillSwitch(monitor)
    kill_switch.kill_file_path = str(tmp_path / "KILL_SWITCH.flag")
    (tmp_path / "KILL_SWITCH.flag").write_text("stop", encoding="utf-8")

    assert kill_switch._check_kill_file()
    assert not (tmp_path / "KILL_SWITCH.flag").exists()

    kill_switch.trigger_kill_switch("manual")

    assert reasons == ["Kill switch: manual"]


def test_kill_switch_start_stop_and_auto_kill(monkeypatch):
    threads: list[_FakeThread] = []
    monitor = monitoring.SafetyMonitor()
    kill_switch = monitoring.KillSwitch(monitor)

    def fake_thread(*, target, daemon=False):
        thread = _FakeThread(target, daemon=daemon)
        threads.append(thread)
        return thread

    monkeypatch.setattr(monitoring.threading, "Thread", fake_thread)
    kill_switch.start_monitoring()
    kill_switch.start_monitoring()
    kill_switch.stop_monitoring()
    kill_switch.set_auto_kill_time(datetime.now(UTC) + timedelta(seconds=5))

    assert len(threads) == 1
    assert threads[0].started
    assert threads[0].joined_timeout == 5
    assert kill_switch.auto_kill_time is not None


def test_performance_monitor_records_and_grades(monkeypatch):
    now = datetime(2026, 4, 24, tzinfo=UTC)
    clock = iter([now, now, now, now + timedelta(hours=1)])
    monkeypatch.setattr(monitoring, "safe_utcnow", lambda: next(clock))

    perf = monitoring.PerformanceMonitor()
    perf.record_order_latency(120.0)
    perf.record_api_response_time("/orders", 80.0)
    perf.record_error("broker")

    report = perf.get_performance_report()

    assert report["average_order_latency_ms"] == 120.0
    assert report["max_order_latency_ms"] == 120.0
    assert report["average_api_response_ms"] == 80.0
    assert report["total_orders_processed"] == 1
    assert report["total_errors"] == 1
    assert report["error_breakdown"] == {"broker": 1}
    assert report["performance_grade"] == "F"


def test_performance_monitor_trims_latency_history():
    perf = monitoring.PerformanceMonitor()

    for idx in range(1001):
        perf.record_order_latency(float(idx))

    latencies = perf.metrics["order_latency"]
    assert len(latencies) == 500
    assert latencies[0]["latency_ms"] == 501.0


def test_file_and_console_alert_callbacks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    alert = {
        "severity": "warning",
        "message": "watch it",
        "timestamp": "2026-04-24T00:00:00Z",
    }

    monitoring.console_alert_callback(alert)
    monitoring.file_alert_callback(alert)

    assert (tmp_path / "trading_alerts.log").read_text(encoding="utf-8").strip()
