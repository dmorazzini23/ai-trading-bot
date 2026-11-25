"""Alert-only watchdog for the trading decision loop.

The scheduler loop in :func:`ai_trading.main.main` calls
:func:`note_decision_loop_progress` each time an iteration finishes (whether
``run_cycle`` returned or raised). It does so in every market state: closed,
halted, safe mode and skipped cycles still finish an iteration. A separate
daemon thread checks the age of the last finished iteration and, during the
regular NYSE session only, reports a stall:

* ``ALERT_DECISION_LOOP_STALLED`` (ERROR, through ``emit_runtime_alert``) when no
  iteration has finished for longer than the threshold, repeated at most once
  per repeat interval while the stall lasts; the journal alert poller turns new
  ERROR lines into operator Slack alerts;
* ``DECISION_LOOP_RECOVERED`` (WARNING) once an iteration finishes again;
* ``decision_loop_stalled`` in the health payload's ``attention_flags``.

The watchdog is alert-only by design: it never halts trading, writes the kill
switch or canary rollback flag, cancels orders, or blocks submissions. A loop
that has stopped cycling cannot act on a halt anyway.

Ages are measured on the monotonic clock; the wall clock is used only for the
market-session check and the reported timestamps.

The threshold is a module constant, not a setting, raised when the loop's own
configured intervals are long: the effective threshold is
``max(900 s, 3 x the largest loop interval)`` passed in when the watchdog is
armed. Market-hours cycle spacing in the service journal (Nov 17-25, 2026,
1,594 gaps) had a median of 94 s and a maximum of 123 s; with the running
intervals (85 s open, 180 s autotune ceiling, 300 s closed) the effective
threshold is 900 s, about 7x the observed worst case.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger
from ai_trading.monitoring.alerts import emit_runtime_alert

logger = get_logger(__name__)

DECISION_LOOP_STALL_THRESHOLD_SECONDS = 900.0
DECISION_LOOP_INTERVAL_MULTIPLIER = 3.0
DECISION_LOOP_STALL_REPEAT_SECONDS = 1800.0
DECISION_LOOP_WATCHDOG_CHECK_SECONDS = 30.0
_MIN_CHECK_SECONDS = 0.01

STALL_EVENT = "ALERT_DECISION_LOOP_STALLED"
RECOVERED_EVENT = "DECISION_LOOP_RECOVERED"
HEALTH_ATTENTION_FLAG = "decision_loop_stalled"

_ET = ZoneInfo("America/New_York")


def _as_utc(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _as_mono(mono: float | None) -> float:
    return float(time.monotonic() if mono is None else mono)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def effective_stall_threshold(loop_interval_seconds: float | None = None) -> float:
    """``max(900 s, 3 x the largest configured loop interval)``."""

    threshold = DECISION_LOOP_STALL_THRESHOLD_SECONDS
    if loop_interval_seconds is not None and float(loop_interval_seconds) > 0.0:
        threshold = max(threshold, DECISION_LOOP_INTERVAL_MULTIPLIER * float(loop_interval_seconds))
    return threshold


class _DecisionLoopWatchdog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._armed_mono: float | None = None
        self._threshold_s = DECISION_LOOP_STALL_THRESHOLD_SECONDS
        self._last_progress_mono: float | None = None
        self._last_progress_at: datetime | None = None
        self._last_cycle_index: int | None = None
        self._stalled = False
        self._stall_baseline_mono: float | None = None
        self._stalled_since_at: datetime | None = None
        self._last_alert_mono: float | None = None
        self._alerts_sent = 0
        self._session_cache: tuple[date, tuple[datetime, datetime] | None] | None = None
        self._calendar_error_day: date | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def _arm_locked(self, *, mono: float, loop_interval_seconds: float | None) -> None:
        self._armed_mono = mono
        self._threshold_s = effective_stall_threshold(loop_interval_seconds)
        self._stalled = False
        self._stall_baseline_mono = None
        self._stalled_since_at = None
        self._last_alert_mono = None
        self._alerts_sent = 0

    def arm(
        self,
        *,
        mono: float | None = None,
        loop_interval_seconds: float | None = None,
    ) -> None:
        with self._lock:
            self._arm_locked(mono=_as_mono(mono), loop_interval_seconds=loop_interval_seconds)

    def note_progress(
        self,
        *,
        cycle_index: int | None = None,
        now: datetime | None = None,
        mono: float | None = None,
    ) -> None:
        stamp = _as_utc(now)
        mono_value = _as_mono(mono)
        with self._lock:
            self._last_progress_mono = mono_value
            self._last_progress_at = stamp
            if cycle_index is not None:
                self._last_cycle_index = int(cycle_index)

    def _fresh_progress_locked(self) -> float | None:
        """Monotonic progress since arming; progress from before arming is not a baseline."""

        armed = self._armed_mono
        progress = self._last_progress_mono
        if armed is None or progress is None or progress < armed:
            return None
        return progress

    def _session_bounds(self, now_utc: datetime) -> tuple[datetime, datetime] | None:
        """Regular-session bounds for the ET day of ``now_utc`` (None when closed all day)."""

        from ai_trading.utils.market_calendar import is_trading_day, rth_session_utc

        session_day = now_utc.astimezone(_ET).date()
        cached = self._session_cache
        if cached is not None and cached[0] == session_day:
            return cached[1]
        bounds: tuple[datetime, datetime] | None = None
        if is_trading_day(session_day):
            bounds = rth_session_utc(session_day)
        self._session_cache = (session_day, bounds)
        return bounds

    def _in_regular_session(self, now_utc: datetime) -> bool:
        try:
            bounds = self._session_bounds(now_utc)
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            session_day = now_utc.astimezone(_ET).date()
            if self._calendar_error_day != session_day:
                self._calendar_error_day = session_day
                logger.warning(
                    "DECISION_LOOP_WATCHDOG_CALENDAR_UNAVAILABLE",
                    extra={"error": str(exc), "error_type": exc.__class__.__name__},
                )
            return False
        if bounds is None:
            return False
        start_utc, end_utc = bounds
        return start_utc <= now_utc < end_utc

    def evaluate(
        self,
        *,
        now: datetime | None = None,
        mono: float | None = None,
    ) -> dict[str, Any] | None:
        """Check once; return the stall alert payload when one was emitted."""

        now_utc = _as_utc(now)
        mono_now = _as_mono(mono)
        with self._lock:
            armed_mono = self._armed_mono
            if armed_mono is None:
                return None
            progress_mono = self._fresh_progress_locked()
            baseline_mono = progress_mono if progress_mono is not None else armed_mono
            threshold_s = self._threshold_s
            stall_baseline = self._stall_baseline_mono
            recovered: dict[str, Any] | None = None
            if self._stalled and stall_baseline is not None and baseline_mono > stall_baseline:
                recovered = {
                    "stalled_seconds": round(baseline_mono - stall_baseline, 1),
                    "last_cycle_index": self._last_cycle_index,
                    "last_progress_ts": _iso(self._last_progress_at),
                    "alerts_sent": self._alerts_sent,
                    "threshold_seconds": threshold_s,
                }
                self._stalled = False
                self._stall_baseline_mono = None
                self._stalled_since_at = None
                self._last_alert_mono = None
                self._alerts_sent = 0
        if recovered is not None:
            logger.warning(RECOVERED_EVENT, extra=recovered)
            return None

        age_seconds = max(0.0, mono_now - baseline_mono)
        if age_seconds <= threshold_s:
            return None
        if not self._in_regular_session(now_utc):
            return None

        with self._lock:
            if not self._stalled:
                self._stalled = True
                self._stall_baseline_mono = baseline_mono
                self._stalled_since_at = (
                    self._last_progress_at if progress_mono is not None else None
                )
            last_alert = self._last_alert_mono
            due = last_alert is None or (mono_now - last_alert) >= DECISION_LOOP_STALL_REPEAT_SECONDS
            if not due:
                return None
            self._last_alert_mono = mono_now
            self._alerts_sent += 1
            details: dict[str, Any] = {
                "seconds_since_progress": round(age_seconds, 1),
                "threshold_seconds": threshold_s,
                "last_cycle_index": self._last_cycle_index,
                "last_progress_ts": _iso(self._last_progress_at) if progress_mono is not None else None,
                "reason": "no_cycle_since_start" if progress_mono is None else "no_cycle_progress",
                "alert_number": self._alerts_sent,
                "repeat_seconds": DECISION_LOOP_STALL_REPEAT_SECONDS,
                "alert_only": True,
            }
        emit_runtime_alert(STALL_EVENT, severity="error", details=details)
        return details

    def snapshot(self, *, mono: float | None = None) -> dict[str, Any]:
        mono_now = _as_mono(mono)
        with self._lock:
            armed_mono = self._armed_mono
            thread = self._thread
            thread_alive = thread is not None and thread.is_alive()
            progress_mono = self._fresh_progress_locked()
            baseline = progress_mono if progress_mono is not None else armed_mono
            age_seconds = round(max(0.0, mono_now - baseline), 1) if baseline is not None else None
            return {
                # Monitored only while armed AND the checking thread is alive.
                "monitored": armed_mono is not None and thread_alive,
                "armed": armed_mono is not None,
                "thread_alive": thread_alive,
                "stalled": bool(self._stalled) and armed_mono is not None,
                "alert_only": True,
                "last_progress_ts": _iso(self._last_progress_at),
                "last_cycle_index": self._last_cycle_index,
                "seconds_since_progress": age_seconds,
                "stall_threshold_seconds": self._threshold_s,
                "stalled_since_ts": _iso(self._stalled_since_at) if self._stalled else None,
                "alerts_sent": self._alerts_sent,
            }

    def _run(self, stop_event: threading.Event, check_seconds: float) -> None:
        while not stop_event.wait(check_seconds):
            try:
                self.evaluate()
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                logger.warning(
                    "DECISION_LOOP_WATCHDOG_CHECK_FAILED",
                    extra={"error": str(exc), "error_type": exc.__class__.__name__},
                )

    def start(
        self,
        *,
        check_seconds: float = DECISION_LOOP_WATCHDOG_CHECK_SECONDS,
        loop_interval_seconds: float | None = None,
    ) -> threading.Thread:
        check_s = max(_MIN_CHECK_SECONDS, float(check_seconds))
        with self._lock:
            # Check, arm, create and start under one lock so racing callers
            # share a single thread. ``_run`` only waits on its event first.
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._arm_locked(mono=_as_mono(None), loop_interval_seconds=loop_interval_seconds)
            threshold_s = self._threshold_s
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run,
                args=(stop_event, check_s),
                name="decision-loop-watchdog",
                daemon=True,
            )
            self._stop_event = stop_event
            self._thread = thread
            thread.start()
        logger.info(
            "DECISION_LOOP_WATCHDOG_STARTED",
            extra={
                "threshold_seconds": threshold_s,
                "loop_interval_seconds": loop_interval_seconds,
                "repeat_seconds": DECISION_LOOP_STALL_REPEAT_SECONDS,
                "check_seconds": check_s,
            },
        )
        return thread

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            self._stop_event = None
            self._thread = None
            self._armed_mono = None
            self._stalled = False
            self._stall_baseline_mono = None
            self._stalled_since_at = None
        if stop_event is not None:
            stop_event.set()


_WATCHDOG = _DecisionLoopWatchdog()


def start_decision_loop_watchdog(
    *,
    check_seconds: float = DECISION_LOOP_WATCHDOG_CHECK_SECONDS,
    loop_interval_seconds: float | None = None,
) -> threading.Thread:
    """Arm the watchdog and start its daemon thread (idempotent).

    ``loop_interval_seconds`` is the largest interval the scheduler loop may
    sleep between iterations; it raises the stall threshold to three times it.
    """

    return _WATCHDOG.start(check_seconds=check_seconds, loop_interval_seconds=loop_interval_seconds)


def stop_decision_loop_watchdog() -> None:
    """Stop the watchdog thread and disarm it (the loop has exited)."""

    _WATCHDOG.stop()


def note_decision_loop_progress(
    *,
    cycle_index: int | None = None,
    now: datetime | None = None,
    mono: float | None = None,
) -> None:
    """Record that the scheduler loop finished an iteration."""

    _WATCHDOG.note_progress(cycle_index=cycle_index, now=now, mono=mono)


def check_decision_loop_stall(
    *,
    now: datetime | None = None,
    mono: float | None = None,
) -> dict[str, Any] | None:
    """Run one watchdog check; return the alert payload if a stall alert was emitted."""

    return _WATCHDOG.evaluate(now=now, mono=mono)


def get_decision_loop_snapshot(*, mono: float | None = None) -> dict[str, Any]:
    """Health view of the watchdog; ``stalled`` is the non-fatal attention flag source."""

    return _WATCHDOG.snapshot(mono=mono)


def _reset_decision_loop_watchdog_for_tests() -> None:
    global _WATCHDOG
    _WATCHDOG.stop()
    _WATCHDOG = _DecisionLoopWatchdog()


__all__ = [
    "DECISION_LOOP_INTERVAL_MULTIPLIER",
    "DECISION_LOOP_STALL_REPEAT_SECONDS",
    "DECISION_LOOP_STALL_THRESHOLD_SECONDS",
    "DECISION_LOOP_WATCHDOG_CHECK_SECONDS",
    "HEALTH_ATTENTION_FLAG",
    "RECOVERED_EVENT",
    "STALL_EVENT",
    "check_decision_loop_stall",
    "effective_stall_threshold",
    "get_decision_loop_snapshot",
    "note_decision_loop_progress",
    "start_decision_loop_watchdog",
    "stop_decision_loop_watchdog",
]
