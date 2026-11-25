from __future__ import annotations

import ast
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ai_trading.monitoring import alerts, decision_loop_watchdog as watchdog
from ai_trading.monitoring import model_liveness

THRESHOLD = watchdog.DECISION_LOOP_STALL_THRESHOLD_SECONDS
REPEAT = watchdog.DECISION_LOOP_STALL_REPEAT_SECONDS
# Wednesday 2026-11-25: regular session 14:30-21:00 UTC.
SESSION_START = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _fresh_watchdog():
    watchdog._reset_decision_loop_watchdog_for_tests()
    yield
    watchdog._reset_decision_loop_watchdog_for_tests()


@pytest.fixture
def alert_calls(monkeypatch) -> list[dict[str, Any]]:
    """Record alerts while still running the real (log-only) emit_runtime_alert."""

    calls: list[dict[str, Any]] = []
    real_emit = alerts.emit_runtime_alert

    def _spy(event: str, **kwargs: Any) -> None:
        calls.append({"event": event, **kwargs})
        real_emit(event, **kwargs)

    monkeypatch.setattr(watchdog, "emit_runtime_alert", _spy)
    return calls


@pytest.fixture
def warnings_logged(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    real_warning = watchdog.logger.warning

    def _spy(msg: str, *args: Any, **kwargs: Any) -> None:
        records.append((msg, dict(kwargs.get("extra") or {})))
        real_warning(msg, *args, **kwargs)

    monkeypatch.setattr(watchdog.logger, "warning", _spy)
    return records


def _arm(at: datetime) -> None:
    watchdog._WATCHDOG.arm(now=at)


def test_stall_detected_during_regular_session(alert_calls) -> None:
    _arm(SESSION_START)
    last = SESSION_START + timedelta(seconds=60)
    watchdog.note_decision_loop_progress(cycle_index=7, now=last)

    assert watchdog.check_decision_loop_stall(now=last + timedelta(seconds=THRESHOLD)) is None
    assert alert_calls == []

    details = watchdog.check_decision_loop_stall(now=last + timedelta(seconds=THRESHOLD + 1))

    assert details is not None
    assert len(alert_calls) == 1
    call = alert_calls[0]
    assert call["event"] == "ALERT_DECISION_LOOP_STALLED"
    assert call["severity"] == "error"
    assert call["details"]["seconds_since_progress"] == pytest.approx(THRESHOLD + 1)
    assert call["details"]["last_cycle_index"] == 7
    assert call["details"]["reason"] == "no_cycle_progress"
    assert call["details"]["alert_only"] is True
    snapshot = watchdog.get_decision_loop_snapshot(now=last + timedelta(seconds=THRESHOLD + 1))
    assert snapshot["stalled"] is True
    assert snapshot["monitored"] is True


@pytest.mark.parametrize(
    "now",
    [
        # Weekday before the open and after the close.
        datetime(2026, 11, 25, 13, 0, tzinfo=UTC),
        datetime(2026, 11, 25, 22, 0, tzinfo=UTC),
        # Saturday.
        datetime(2026, 11, 28, 17, 0, tzinfo=UTC),
        # Thanksgiving holiday.
        datetime(2026, 11, 26, 17, 0, tzinfo=UTC),
        # Black Friday half day closes 13:00 ET (18:00 UTC).
        datetime(2026, 11, 27, 18, 30, tzinfo=UTC),
    ],
    ids=["pre_open", "after_close", "weekend", "holiday", "half_day_afternoon"],
)
def test_no_alert_outside_regular_session(alert_calls, now: datetime) -> None:
    _arm(now - timedelta(hours=6))

    assert watchdog.check_decision_loop_stall(now=now) is None
    assert alert_calls == []
    assert watchdog.get_decision_loop_snapshot(now=now)["stalled"] is False


def test_half_day_morning_still_alerts(alert_calls) -> None:
    now = datetime(2026, 11, 27, 17, 0, tzinfo=UTC)
    _arm(now - timedelta(seconds=THRESHOLD + 60))

    assert watchdog.check_decision_loop_stall(now=now) is not None
    assert len(alert_calls) == 1


def test_no_alert_during_startup_warmup(alert_calls) -> None:
    _arm(SESSION_START)
    # Progress recorded before arming (warm-up, an earlier run) is not a baseline.
    watchdog.note_decision_loop_progress(cycle_index=0, now=SESSION_START - timedelta(hours=2))

    assert watchdog.check_decision_loop_stall(now=SESSION_START + timedelta(seconds=600)) is None
    assert watchdog.check_decision_loop_stall(now=SESSION_START + timedelta(seconds=THRESHOLD)) is None
    assert alert_calls == []

    details = watchdog.check_decision_loop_stall(now=SESSION_START + timedelta(seconds=THRESHOLD + 5))
    assert details is not None
    assert details["reason"] == "no_cycle_since_start"
    assert details["last_progress_ts"] is None


def test_unarmed_watchdog_never_alerts(alert_calls) -> None:
    watchdog.note_decision_loop_progress(cycle_index=1, now=SESSION_START)

    assert watchdog.check_decision_loop_stall(now=SESSION_START + timedelta(hours=3)) is None
    assert alert_calls == []
    assert watchdog.get_decision_loop_snapshot()["monitored"] is False


def test_repeat_interval_limits_alerts(alert_calls) -> None:
    _arm(SESSION_START)
    first = SESSION_START + timedelta(seconds=THRESHOLD + 1)

    assert watchdog.check_decision_loop_stall(now=first) is not None
    for offset in (30, 60, REPEAT - 1):
        assert watchdog.check_decision_loop_stall(now=first + timedelta(seconds=offset)) is None
    second = watchdog.check_decision_loop_stall(now=first + timedelta(seconds=REPEAT))

    assert second is not None
    assert second["alert_number"] == 2
    assert [call["details"]["alert_number"] for call in alert_calls] == [1, 2]


def test_recovery_logged_once_progress_resumes(alert_calls, warnings_logged) -> None:
    _arm(SESSION_START)
    watchdog.note_decision_loop_progress(cycle_index=3, now=SESSION_START)
    stalled_at = SESSION_START + timedelta(seconds=THRESHOLD + 1)
    assert watchdog.check_decision_loop_stall(now=stalled_at) is not None

    resumed = stalled_at + timedelta(seconds=120)
    watchdog.note_decision_loop_progress(cycle_index=4, now=resumed)
    assert watchdog.check_decision_loop_stall(now=resumed + timedelta(seconds=30)) is None

    recovered = [extra for msg, extra in warnings_logged if msg == "DECISION_LOOP_RECOVERED"]
    assert len(recovered) == 1
    assert recovered[0]["last_cycle_index"] == 4
    assert recovered[0]["stalled_seconds"] == pytest.approx(THRESHOLD + 121)
    assert recovered[0]["alerts_sent"] == 1
    snapshot = watchdog.get_decision_loop_snapshot(now=resumed + timedelta(seconds=30))
    assert snapshot["stalled"] is False
    assert snapshot["alerts_sent"] == 0

    # Healthy cycling afterwards: no further recovery lines, no alerts.
    for step in range(1, 4):
        tick = resumed + timedelta(seconds=95 * step)
        watchdog.note_decision_loop_progress(cycle_index=4 + step, now=tick)
        assert watchdog.check_decision_loop_stall(now=tick + timedelta(seconds=30)) is None
    assert len(alert_calls) == 1
    assert sum(1 for msg, _ in warnings_logged if msg == "DECISION_LOOP_RECOVERED") == 1

    # A new stall alerts again, numbered from one.
    later = resumed + timedelta(seconds=95 * 3 + THRESHOLD + 1)
    again = watchdog.check_decision_loop_stall(now=later)
    assert again is not None
    assert again["alert_number"] == 1


def test_stall_path_is_alert_only(monkeypatch, tmp_path, alert_calls) -> None:
    """The stall path only logs: no halt, kill switch, canary rollback, cancel or submit."""

    data_dir = tmp_path / "alert_only_data"
    data_dir.mkdir()
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    forbidden: list[str] = []

    def _record(name: str):
        def _call(*_args: Any, **_kwargs: Any) -> None:
            forbidden.append(name)

        return _call

    monkeypatch.setattr(model_liveness, "maybe_trigger_canary_auto_rollback", _record("rollback"))
    monkeypatch.setattr(
        model_liveness._MONITOR, "maybe_trigger_canary_rollback", _record("rollback_monitor")
    )
    monkeypatch.setattr(model_liveness, "_atomic_write_text", _record("kill_switch_or_flag_write"))
    monkeypatch.setattr(subprocess, "run", _record("subprocess"))
    monkeypatch.setattr(os, "replace", _record("file_replace"))

    _arm(SESSION_START)
    first = SESSION_START + timedelta(seconds=THRESHOLD + 1)
    assert watchdog.check_decision_loop_stall(now=first) is not None
    assert watchdog.check_decision_loop_stall(now=first + timedelta(seconds=REPEAT)) is not None

    assert forbidden == []
    assert len(alert_calls) == 2
    for call in alert_calls:
        # emit_runtime_alert only sets a halt reason when given state + halt_reason.
        assert call.get("state") is None
        assert call.get("halt_reason") is None
    # No kill switch, canary rollback flag or any other runtime file was written.
    assert [p for p in Path(tmp_path).rglob("*") if p.is_file()] == []

    # Statically: the module imports nothing that can act on trading.
    source = Path(watchdog.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(str(node.module))
    assert imported <= {
        "__future__",
        "threading",
        "datetime",
        "typing",
        "zoneinfo",
        "ai_trading.exception_family",
        "ai_trading.logging",
        "ai_trading.monitoring.alerts",
        "ai_trading.utils.market_calendar",
    }
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for token in ("halt", "kill", "rollback", "cancel", "submit", "request_stop", "write"):
        assert not [name for name in identifiers if token in name.lower()], token


def test_watchdog_thread_runs_off_loop_and_stops(alert_calls) -> None:
    thread = watchdog.start_decision_loop_watchdog(check_seconds=1.0)
    try:
        assert thread.name == "decision-loop-watchdog"
        assert thread.daemon is True
        assert thread.is_alive()
        assert watchdog.start_decision_loop_watchdog(check_seconds=1.0) is thread
        assert watchdog.get_decision_loop_snapshot()["monitored"] is True
    finally:
        watchdog.stop_decision_loop_watchdog()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert watchdog.get_decision_loop_snapshot()["monitored"] is False
    assert alert_calls == []


def test_calendar_failure_does_not_alert_or_raise(monkeypatch, alert_calls, warnings_logged) -> None:
    from ai_trading.utils import market_calendar

    def _broken(_day):
        raise RuntimeError("calendar down")

    monkeypatch.setattr(market_calendar, "is_trading_day", _broken)
    _arm(SESSION_START)
    now = SESSION_START + timedelta(seconds=THRESHOLD + 1)

    assert watchdog.check_decision_loop_stall(now=now) is None
    assert watchdog.check_decision_loop_stall(now=now + timedelta(seconds=30)) is None
    assert alert_calls == []
    assert [msg for msg, _ in warnings_logged] == ["DECISION_LOOP_WATCHDOG_CALENDAR_UNAVAILABLE"]
