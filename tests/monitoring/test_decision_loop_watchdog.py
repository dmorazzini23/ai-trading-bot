from __future__ import annotations

import ast
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ai_trading.monitoring import alerts, decision_loop_watchdog as watchdog

THRESHOLD = watchdog.DECISION_LOOP_STALL_THRESHOLD_SECONDS
REPEAT = watchdog.DECISION_LOOP_STALL_REPEAT_SECONDS
# Wednesday 2026-11-25: regular session 14:30-21:00 UTC.
SESSION_START = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _m(at: datetime) -> float:
    """Deterministic monotonic reading that tracks the wall clock in these tests."""

    return (at - _EPOCH).total_seconds()


def _arm(at: datetime, *, loop_interval_seconds: float | None = None) -> None:
    watchdog._WATCHDOG.arm(mono=_m(at), loop_interval_seconds=loop_interval_seconds)


def _progress(at: datetime, cycle_index: int) -> None:
    watchdog.note_decision_loop_progress(cycle_index=cycle_index, now=at, mono=_m(at))


def _check(at: datetime) -> dict[str, Any] | None:
    return watchdog.check_decision_loop_stall(now=at, mono=_m(at))


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


def test_stall_detected_during_regular_session(alert_calls) -> None:
    _arm(SESSION_START)
    last = SESSION_START + timedelta(seconds=60)
    _progress(last, 7)

    assert _check(last + timedelta(seconds=THRESHOLD)) is None
    assert alert_calls == []

    details = _check(last + timedelta(seconds=THRESHOLD + 1))

    assert details is not None
    assert len(alert_calls) == 1
    call = alert_calls[0]
    assert call["event"] == "ALERT_DECISION_LOOP_STALLED"
    assert call["severity"] == "error"
    assert call["details"]["seconds_since_progress"] == pytest.approx(THRESHOLD + 1)
    assert call["details"]["threshold_seconds"] == THRESHOLD
    assert call["details"]["last_cycle_index"] == 7
    assert call["details"]["reason"] == "no_cycle_progress"
    assert call["details"]["alert_only"] is True
    snapshot = watchdog.get_decision_loop_snapshot(mono=_m(last + timedelta(seconds=THRESHOLD + 1)))
    assert snapshot["stalled"] is True
    assert snapshot["armed"] is True
    assert snapshot["stall_threshold_seconds"] == THRESHOLD


@pytest.mark.parametrize(
    ("loop_interval", "expected"),
    [(None, 900.0), (85.0, 900.0), (300.0, 900.0), (600.0, 1800.0), (1200.0, 3600.0)],
)
def test_effective_threshold_scales_with_loop_interval(loop_interval, expected) -> None:
    assert watchdog.effective_stall_threshold(loop_interval) == expected


def test_long_configured_interval_does_not_false_alert(alert_calls) -> None:
    # A 600 s closed-market interval: the first open cycle can be ~650 s after
    # the last closed one. Threshold becomes 1800 s, so no alert at 1000 s.
    _arm(SESSION_START - timedelta(hours=1), loop_interval_seconds=600.0)
    _progress(SESSION_START - timedelta(seconds=400), 1)

    assert _check(SESSION_START + timedelta(seconds=600)) is None
    assert alert_calls == []
    details = _check(SESSION_START + timedelta(seconds=1401))
    assert details is not None
    assert details["threshold_seconds"] == 1800.0
    assert watchdog.get_decision_loop_snapshot()["stall_threshold_seconds"] == 1800.0


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

    assert _check(now) is None
    assert alert_calls == []
    assert watchdog.get_decision_loop_snapshot(mono=_m(now))["stalled"] is False


def test_half_day_morning_still_alerts(alert_calls) -> None:
    now = datetime(2026, 11, 27, 17, 0, tzinfo=UTC)
    _arm(now - timedelta(seconds=THRESHOLD + 60))

    assert _check(now) is not None
    assert len(alert_calls) == 1


def test_no_alert_during_startup_warmup(alert_calls) -> None:
    _arm(SESSION_START)
    # Progress recorded before arming (warm-up, an earlier run) is not a baseline.
    _progress(SESSION_START - timedelta(hours=2), 0)

    assert _check(SESSION_START + timedelta(seconds=600)) is None
    assert _check(SESSION_START + timedelta(seconds=THRESHOLD)) is None
    assert alert_calls == []

    details = _check(SESSION_START + timedelta(seconds=THRESHOLD + 5))
    assert details is not None
    assert details["reason"] == "no_cycle_since_start"
    assert details["last_progress_ts"] is None


def test_wall_clock_step_with_healthy_loop_does_not_alert(alert_calls) -> None:
    """A forward wall-clock jump is not elapsed time: ages use the monotonic clock."""

    _arm(SESSION_START)
    mono = _m(SESSION_START)
    for step in range(1, 6):
        mono += 95.0
        # Wall clock jumps two hours forward after the second cycle.
        wall = SESSION_START + timedelta(seconds=95 * step) + (
            timedelta(hours=2) if step > 2 else timedelta(0)
        )
        watchdog.note_decision_loop_progress(cycle_index=step, now=wall, mono=mono)
        assert watchdog.check_decision_loop_stall(now=wall + timedelta(seconds=30), mono=mono + 30.0) is None
    assert alert_calls == []
    snapshot = watchdog.get_decision_loop_snapshot(mono=mono + 30.0)
    assert snapshot["seconds_since_progress"] == pytest.approx(30.0)
    assert snapshot["stalled"] is False


def test_unarmed_watchdog_never_alerts(alert_calls) -> None:
    _progress(SESSION_START, 1)

    assert _check(SESSION_START + timedelta(hours=3)) is None
    assert alert_calls == []
    snapshot = watchdog.get_decision_loop_snapshot()
    assert snapshot["monitored"] is False
    assert snapshot["armed"] is False


def test_repeat_interval_limits_alerts(alert_calls) -> None:
    _arm(SESSION_START)
    first = SESSION_START + timedelta(seconds=THRESHOLD + 1)

    assert _check(first) is not None
    for offset in (30, 60, REPEAT - 1):
        assert _check(first + timedelta(seconds=offset)) is None
    second = _check(first + timedelta(seconds=REPEAT))

    assert second is not None
    assert second["alert_number"] == 2
    assert [call["details"]["alert_number"] for call in alert_calls] == [1, 2]


def test_recovery_logged_once_progress_resumes(alert_calls, warnings_logged) -> None:
    _arm(SESSION_START)
    _progress(SESSION_START, 3)
    stalled_at = SESSION_START + timedelta(seconds=THRESHOLD + 1)
    assert _check(stalled_at) is not None

    resumed = stalled_at + timedelta(seconds=120)
    _progress(resumed, 4)
    assert _check(resumed + timedelta(seconds=30)) is None

    recovered = [extra for msg, extra in warnings_logged if msg == "DECISION_LOOP_RECOVERED"]
    assert len(recovered) == 1
    assert recovered[0]["last_cycle_index"] == 4
    assert recovered[0]["stalled_seconds"] == pytest.approx(THRESHOLD + 121)
    assert recovered[0]["alerts_sent"] == 1
    snapshot = watchdog.get_decision_loop_snapshot(mono=_m(resumed + timedelta(seconds=30)))
    assert snapshot["stalled"] is False
    assert snapshot["alerts_sent"] == 0

    # Healthy cycling afterwards: no further recovery lines, no alerts.
    for step in range(1, 4):
        tick = resumed + timedelta(seconds=95 * step)
        _progress(tick, 4 + step)
        assert _check(tick + timedelta(seconds=30)) is None
    assert len(alert_calls) == 1
    assert sum(1 for msg, _ in warnings_logged if msg == "DECISION_LOOP_RECOVERED") == 1

    # A new stall alerts again, numbered from one.
    later = resumed + timedelta(seconds=95 * 3 + THRESHOLD + 1)
    again = _check(later)
    assert again is not None
    assert again["alert_number"] == 1


def test_stall_path_is_alert_only(alert_calls) -> None:
    """The stall path only logs: no halt, kill switch, canary rollback, cancel or submit."""

    _arm(SESSION_START)
    first = SESSION_START + timedelta(seconds=THRESHOLD + 1)
    assert _check(first) is not None
    assert _check(first + timedelta(seconds=REPEAT)) is not None

    assert len(alert_calls) == 2
    for call in alert_calls:
        # emit_runtime_alert only sets a halt reason when given state + halt_reason.
        assert call.get("state") is None
        assert call.get("halt_reason") is None

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
        "time",
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


def test_watchdog_thread_runs_evaluate_off_loop_and_stops(monkeypatch, alert_calls) -> None:
    called = threading.Event()
    caller_names: list[str] = []
    instance = watchdog._WATCHDOG
    real_evaluate = instance.evaluate

    def _evaluate(**kwargs: Any) -> dict[str, Any] | None:
        caller_names.append(threading.current_thread().name)
        result = real_evaluate(**kwargs)
        called.set()
        return result

    monkeypatch.setattr(instance, "evaluate", _evaluate)
    thread = watchdog.start_decision_loop_watchdog(check_seconds=0.01, loop_interval_seconds=85.0)
    try:
        assert called.wait(timeout=2.0)
        assert thread.name == "decision-loop-watchdog"
        assert thread.daemon is True
        assert caller_names[0] == "decision-loop-watchdog"
        assert threading.current_thread().name not in caller_names
        snapshot = watchdog.get_decision_loop_snapshot()
        assert snapshot["monitored"] is True
        assert snapshot["thread_alive"] is True
        assert snapshot["stall_threshold_seconds"] == THRESHOLD
    finally:
        watchdog.stop_decision_loop_watchdog()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert watchdog.get_decision_loop_snapshot()["monitored"] is False
    assert alert_calls == []


def test_run_logs_check_failure_and_keeps_running(monkeypatch, warnings_logged) -> None:
    second_call = threading.Event()
    calls = {"n": 0}
    instance = watchdog._WATCHDOG

    def _evaluate(**_kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("check exploded")
        second_call.set()

    monkeypatch.setattr(instance, "evaluate", _evaluate)
    thread = watchdog.start_decision_loop_watchdog(check_seconds=0.01)
    try:
        assert second_call.wait(timeout=2.0)
        assert thread.is_alive()
    finally:
        watchdog.stop_decision_loop_watchdog()
    thread.join(timeout=2.0)
    failures = [extra for msg, extra in warnings_logged if msg == "DECISION_LOOP_WATCHDOG_CHECK_FAILED"]
    assert failures and failures[0]["error"] == "check exploded"
    assert failures[0]["error_type"] == "RuntimeError"


def test_dead_watchdog_thread_reports_not_monitored() -> None:
    dead = threading.Thread(target=lambda: None, name="decision-loop-watchdog")
    dead.start()
    dead.join()
    instance = watchdog._WATCHDOG
    instance.arm()
    instance._thread = dead

    snapshot = watchdog.get_decision_loop_snapshot()

    assert snapshot["armed"] is True
    assert snapshot["thread_alive"] is False
    assert snapshot["monitored"] is False


def test_concurrent_start_creates_one_thread() -> None:
    barrier = threading.Barrier(8)
    started: list[threading.Thread] = []
    lock = threading.Lock()

    def _start() -> None:
        barrier.wait()
        thread = watchdog.start_decision_loop_watchdog(check_seconds=60.0)
        with lock:
            started.append(thread)

    racers = [threading.Thread(target=_start) for _ in range(8)]
    for racer in racers:
        racer.start()
    for racer in racers:
        racer.join(timeout=5.0)
    try:
        assert len(started) == 8
        assert len({id(thread) for thread in started}) == 1
        alive = [t for t in threading.enumerate() if t.name == "decision-loop-watchdog"]
        assert alive == [started[0]]
    finally:
        watchdog.stop_decision_loop_watchdog()
    started[0].join(timeout=2.0)
    assert not started[0].is_alive()


def test_calendar_failure_does_not_alert_or_raise(monkeypatch, alert_calls, warnings_logged) -> None:
    from ai_trading.utils import market_calendar

    def _broken(_day):
        raise RuntimeError("calendar down")

    monkeypatch.setattr(market_calendar, "is_trading_day", _broken)
    _arm(SESSION_START)
    now = SESSION_START + timedelta(seconds=THRESHOLD + 1)

    assert _check(now) is None
    assert _check(now + timedelta(seconds=30)) is None
    assert alert_calls == []
    assert [msg for msg, _ in warnings_logged] == ["DECISION_LOOP_WATCHDOG_CALENDAR_UNAVAILABLE"]
