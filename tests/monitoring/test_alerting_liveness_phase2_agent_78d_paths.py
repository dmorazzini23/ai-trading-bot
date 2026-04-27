from __future__ import annotations

import queue
import smtplib
import sys
import types
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_trading.monitoring import alerting
from ai_trading.monitoring import model_liveness
from ai_trading.monitoring import performance_dashboard as dashboard


class _BadContext:
    def __enter__(self) -> None:
        raise TypeError("lock unavailable")

    def __exit__(self, *_args: object) -> None:
        return None


def _set_liveness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_REQUIRE_MARKET_OPEN", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "0")
    monkeypatch.setenv("USE_RL_AGENT", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "SPY")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "0")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COMMAND", "")
    model_liveness._reset_model_liveness_state_for_tests()


def test_email_and_slack_configuration_success_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[tuple[object, ...]] = []

    class SMTP:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> SMTP:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def starttls(self) -> None:
            return None

        def login(self, *_args: object) -> None:
            return None

        def send_message(self, _message: object) -> None:
            raise smtplib.SMTPException("smtp refused")

    monkeypatch.setattr(alerting.smtplib, "SMTP", SMTP)
    email = alerting.EmailAlerter("smtp.example.com", 2525, "bot", "secret")
    sent = email.send_alert(
        alerting.Alert("Margin", "threshold crossed", alerting.AlertSeverity.CRITICAL),
        ["ops@example.com"],
    )
    assert sent is False

    manager = alerting.AlertManager()
    manager.configure_email("smtp.example.com", 2525, "bot", "secret", ["ops@example.com"])
    assert manager.email_recipients == ["ops@example.com"]
    manager.configure_slack("https://hooks.example/slack", "#ops")
    assert manager.slack_alerter.channel == "#ops"

    def bad_email(*_args: object, **_kwargs: object) -> object:
        raise ValueError("bad email")

    def bad_slack(*_args: object, **_kwargs: object) -> object:
        raise TypeError("bad slack")

    monkeypatch.setattr(alerting, "EmailAlerter", bad_email)
    manager.configure_email("smtp.example.com", 2525, "bot", "secret", [])
    monkeypatch.setattr(alerting, "SlackAlerter", bad_slack)
    manager.configure_slack("https://hooks.example/slack")
    configured.append((manager.email_recipients, manager.slack_alerter.channel))
    assert configured


def test_alert_manager_defensive_and_transition_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = alerting.AlertManager()
    manager.is_running = True
    manager.start_processing()

    class BadThread:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("thread unavailable")

    manager.is_running = False
    monkeypatch.setattr(alerting.threading, "Thread", BadThread)
    manager.start_processing()

    class BadAlive:
        def is_alive(self) -> bool:
            raise RuntimeError("join state unavailable")

    manager.processing_thread = BadAlive()
    manager.stop_processing()

    def bad_alert(*_args: object, **_kwargs: object) -> object:
        raise ValueError("invalid alert")

    monkeypatch.setattr(alerting, "Alert", bad_alert)
    assert manager.send_alert("bad", "bad") == ""


def test_alert_manager_processing_error_and_channel_fallbacks() -> None:
    manager = alerting.AlertManager()
    handled: list[str] = []
    alert = alerting.Alert("Handler", "boom", alerting.AlertSeverity.WARNING)
    manager.add_custom_handler(
        alerting.AlertSeverity.WARNING,
        lambda _alert: (_ for _ in ()).throw(RuntimeError("handler failed")),
    )
    manager.is_running = True
    manager.alert_queue.put(alert)

    def stop_after_send(_alert: alerting.Alert, _channel: alerting.AlertChannel) -> bool:
        handled.append("sent")
        manager.is_running = False
        return False

    manager._send_to_channel = stop_after_send  # type: ignore[method-assign]
    manager._process_alerts()
    assert handled == ["sent", "sent"]

    class EmptyOnce:
        def get(self, timeout: float) -> object:
            assert timeout == 1.0
            manager.is_running = False
            raise queue.Empty

    manager.alert_queue = EmptyOnce()  # type: ignore[assignment]
    manager.is_running = True
    manager._process_alerts()

    class BadGet:
        def get(self, timeout: float) -> object:
            assert timeout == 1.0
            manager.is_running = False
            raise AttributeError("bad queue")

    manager.alert_queue = BadGet()  # type: ignore[assignment]
    manager.is_running = True
    manager._process_alerts()

    class BadBool:
        def __bool__(self) -> bool:
            raise RuntimeError("loop state unavailable")

    manager.is_running = BadBool()  # type: ignore[assignment]
    manager._process_alerts()

    manager = alerting.AlertManager()
    manager.email_alerter.send_alert = lambda *_args: True  # type: ignore[method-assign]
    assert manager._send_to_channel(alert, alerting.AlertChannel.EMAIL) is True
    assert manager._send_to_channel(alert, alerting.AlertChannel.SMS) is False
    assert manager._send_to_channel(alert, alerting.AlertChannel.DISCORD) is False

    def bad_send(*_args: object) -> bool:
        raise RuntimeError("send failed")

    manager.slack_alerter.send_alert = bad_send  # type: ignore[method-assign]
    assert manager._send_to_channel(alert, alerting.AlertChannel.SLACK) is False


def test_alert_manager_rate_stats_and_message_error_fallbacks() -> None:
    manager = alerting.AlertManager()
    emergency = alerting.Alert("Panic", "now", alerting.AlertSeverity.EMERGENCY)
    assert manager._is_rate_limited(emergency) is False

    bad_recent = types.SimpleNamespace(
        timestamp="not a datetime",
        severity=alerting.AlertSeverity.WARNING,
        title="Bad",
    )
    manager.alert_history = [bad_recent]
    assert manager._is_rate_limited(
        alerting.Alert("Bad", "bad", alerting.AlertSeverity.WARNING),
    ) is False

    manager.alert_history = []
    manager.max_history_size = 1
    manager._update_alert_history(alerting.Alert("A", "a", alerting.AlertSeverity.INFO))
    manager._update_alert_history(alerting.Alert("B", "b", alerting.AlertSeverity.INFO))
    assert [item.title for item in manager.alert_history] == ["B"]

    manager.alert_history = None  # type: ignore[assignment]
    manager._update_alert_history(alerting.Alert("C", "c", alerting.AlertSeverity.INFO))
    assert "error" in manager.get_alert_stats()

    class BadDetails:
        def __bool__(self) -> bool:
            return True

        def items(self) -> object:
            raise ValueError("details failed")

    manager = alerting.AlertManager()
    manager.send_alert = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("send"))
    assert manager.send_trading_alert("Fill", "SPY", BadDetails()) == ""  # type: ignore[arg-type]
    assert manager.send_system_alert("OMS", "DOWN") == ""
    assert manager.send_performance_alert("latency", 2, 1) == ""


def test_liveness_env_parse_metric_defaults_and_disabled_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object | None]] = []

    def fake_get_env(name: str, default: object = None, *, cast: object | None = None) -> object:
        calls.append((name, cast))
        if cast is not None:
            raise ValueError("cast failed")
        if name == "BOOL":
            return "yes"
        if name == "FLOAT":
            return "7.5"
        raise RuntimeError("raw missing")

    monkeypatch.setattr(model_liveness, "get_env", fake_get_env)
    assert model_liveness._env_bool("BOOL", False) is True
    assert model_liveness._env_float("FLOAT", 1.0) == 7.5
    assert model_liveness._env_float("MISSING", 2.0) == 2.0
    assert calls

    assert model_liveness._parse_rollback_command("") == []
    assert model_liveness._parse_rollback_command("echo rollback") == ["echo", "rollback"]
    with pytest.raises(ValueError, match="list is empty"):
        model_liveness._parse_rollback_command("[]")
    with pytest.raises(ValueError, match="command is empty"):
        model_liveness._parse_rollback_command('""')
    assert model_liveness._event_for_metric("custom_metric") == "custom_metric"

    monkeypatch.setattr(model_liveness, "get_env", lambda *_args, **_kwargs: False)
    assert model_liveness._ml_liveness_expected_default() is False

    def enabled_get_env(name: str, default: object = None, **_kwargs: object) -> object:
        if name == "AI_TRADING_MODEL_LIVENESS_ENFORCE_ML":
            return True
        return default

    monkeypatch.setattr(model_liveness, "get_env", enabled_get_env)
    monkeypatch.delitem(sys.modules, "ai_trading.core.bot_engine", raising=False)
    assert model_liveness._ml_liveness_expected_default() is True
    module = types.ModuleType("ai_trading.core.bot_engine")
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", module)
    assert model_liveness._ml_liveness_expected_default() is True

    from ai_trading.config.management import get_env as config_get_env

    monkeypatch.setattr(model_liveness, "get_env", config_get_env)
    _set_liveness_env(monkeypatch)
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENABLED", "0")
    assert model_liveness.check_model_liveness(market_open=True) == []


def test_liveness_naive_timestamps_stale_reason_snapshot_and_rollbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _set_liveness_env(monkeypatch)
    now = datetime(2026, 4, 27, 14, 0)
    model_liveness.note_ml_signal(now=now - timedelta(seconds=5))
    breaches = model_liveness.check_model_liveness(
        market_open=True,
        ml_expected=True,
        rl_expected=False,
        now=now,
    )
    assert breaches[0]["reason"] == "stale"
    assert breaches[0]["last_ml_signal_ts"].endswith("+00:00")

    snapshot = model_liveness._MONITOR.snapshot(now=now + timedelta(seconds=1))
    assert snapshot["ml_age_s"] == 6.0

    payload = [
        {
            "metric": "ml_signal",
            "event": "ML_SIGNAL",
            "age_seconds": 5.0,
            "threshold_seconds": 1.0,
            "severity": "warning",
            "reason": "stale",
        }
    ]
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "0")
    assert model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now) is None
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "0")
    assert model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now) is None
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "")
    assert model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now) is None
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "SPY")
    assert (
        model_liveness.maybe_trigger_canary_auto_rollback(
            [{**payload[0], "metric": "after_hours_training"}],
            now=now,
        )
        is None
    )

    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_FLAG_PATH", str(tmp_path / "flag"))
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH_PATH", str(tmp_path / "kill"))
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "1")

    def bad_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(model_liveness, "_atomic_write_text", bad_write)
    result = model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now)
    assert result is not None
    assert result["triggered"] is True
    assert "flag_path" not in result
    assert "kill_switch_path" not in result

    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COMMAND", "[]")
    model_liveness._reset_model_liveness_state_for_tests()
    monkeypatch.setattr(model_liveness, "_atomic_write_text", lambda *_args, **_kwargs: None)
    result = model_liveness.maybe_trigger_canary_auto_rollback(
        [payload[0], {"metric": object()}],
        now=now,
    )
    assert result is not None
    assert result["command_error"]

    class BadEntry:
        def get(self, _key: str) -> object:
            raise ValueError("bad payload")

    assert model_liveness.maybe_trigger_canary_auto_rollback([BadEntry()], now=now) is None


def test_liveness_atomic_write_cleans_failed_tempfile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    removed: list[str] = []

    def bad_replace(_src: str, _dst: model_liveness.Path) -> None:
        raise OSError("replace failed")

    def bad_unlink(path: str) -> None:
        removed.append(path)
        raise OSError("unlink failed")

    monkeypatch.setattr(model_liveness.os, "replace", bad_replace)
    monkeypatch.setattr(model_liveness.os, "unlink", bad_unlink)

    with pytest.raises(OSError, match="replace failed"):
        model_liveness._atomic_write_text(tmp_path / "rollback.flag", "rollback\n")

    assert removed


def test_dashboard_metric_error_paths_and_noop_thresholds() -> None:
    class BadAppend:
        def append(self, _value: object) -> None:
            raise ValueError("append failed")

    metrics = dashboard.PerformanceMetrics()
    metrics.returns = BadAppend()  # type: ignore[assignment]
    metrics.add_return(0.01, 100.0)
    metrics.add_trade("SPY", "bad", datetime.now(UTC), 1.0, 2.0, 1, 1.0)  # type: ignore[arg-type]
    assert metrics.calculate_sharpe_ratio([object()] * 30) == 0.0  # type: ignore[list-item]
    assert metrics.calculate_sortino_ratio([0.01] * 29) == 0.0
    assert metrics.calculate_sortino_ratio([-0.01] * 30) == 0.0
    assert metrics.calculate_sortino_ratio([object()] * 30) == 0.0  # type: ignore[list-item]
    assert metrics.calculate_max_drawdown([None, 1.0]) == (0.0, 0)  # type: ignore[list-item]
    metrics.trades.append({})
    assert metrics.calculate_win_rate()["win_rate"] == 0.0

    metrics = dashboard.PerformanceMetrics()
    metrics._calculate_metrics()
    for idx in range(30):
        metrics.returns.append(0.01 if idx % 2 else -0.01)
        metrics.equity_curve.append(100.0 + idx)
    metrics.calculate_win_rate = lambda: {}  # type: ignore[method-assign]
    metrics._calculate_metrics()

    pnl = dashboard.RealTimePnLTracker()
    pnl.positions = None  # type: ignore[assignment]
    pnl.update_position("SPY", 1, 1.0, 1.0)
    pnl.record_trade("SPY", -1, 1.0, 0.0, "sell")
    pnl._lock = _BadContext()  # type: ignore[assignment]
    pnl.start_new_session(100.0)
    pnl.update_equity(99.0)
    assert "error" in pnl.get_pnl_summary()
    assert pnl.get_position_details() == []

    pnl = dashboard.RealTimePnLTracker()
    pnl.positions = {"bad": {}}
    pnl._calculate_unrealized_pnl()
    assert pnl.unrealized_pnl == 0.0

    detector = dashboard.AnomalyDetector()
    detector.returns_history = BadAppend()  # type: ignore[assignment]
    detector.update_data(0.0, 0.0, 0.0)
    detector.returns_history = [object()] * 30  # type: ignore[assignment,list-item]
    detector.pnl_history.extend([1.0] * 30)
    detector.volatility_history.extend([0.1] * 20)
    assert detector.detect_anomalies(0.0, 0.0, 0.0) == []
    detector._update_thresholds()
    detector.recent_anomalies.append({})
    assert detector.get_recent_anomalies() == []

    perf = dashboard.PerformanceDashboard()
    perf.update_performance(100.0)
    perf.metrics.add_return = lambda *_args: (_ for _ in ()).throw(ValueError("bad metric"))  # type: ignore[method-assign]
    perf.update_performance(100.0, daily_return=0.01)


def test_dashboard_alert_summary_and_forwarding_error_paths() -> None:
    class AlertManager:
        def __init__(self) -> None:
            self.alerts: list[tuple[object, ...]] = []

        def send_performance_alert(self, *args: object) -> None:
            self.alerts.append(args)

    manager = AlertManager()
    perf = dashboard.PerformanceDashboard(manager)  # type: ignore[arg-type]
    perf.metrics.current_metrics = {
        "sharpe_ratio": -10.0,
        "max_drawdown": 1.0,
        "win_rate": 0.0,
    }
    perf._check_performance_thresholds()
    assert [entry[0] for entry in manager.alerts] == [
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Win Rate",
    ]

    perf.metrics.get_current_metrics = lambda: (_ for _ in ()).throw(ValueError("metrics"))  # type: ignore[method-assign]
    perf._check_performance_thresholds()

    perf.pnl_tracker.get_pnl_summary = lambda: (_ for _ in ()).throw(ValueError("summary"))  # type: ignore[method-assign]
    assert "error" in perf.get_dashboard_summary()

    perf.metrics.add_trade = lambda *_args: (_ for _ in ()).throw(ValueError("trade"))  # type: ignore[method-assign]
    perf.add_trade("SPY", datetime.now(UTC), datetime.now(UTC), 1.0, 2.0, 1, 1.0)

    perf.pnl_tracker.update_position = lambda *_args: (_ for _ in ()).throw(ValueError("position"))  # type: ignore[method-assign]
    perf.update_position("SPY", 1, 1.0, 2.0)
