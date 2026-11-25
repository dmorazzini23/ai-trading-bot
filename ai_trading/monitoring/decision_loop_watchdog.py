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

Thresholds are module constants, not settings. Market-hours cycle spacing in
the service journal (Nov 17-25, 2026, 1,594 gaps) had a median of 94 s and a
maximum of 123 s; the interval autotune ceiling is 180 s and the closed-market
interval 300 s, so 900 s is about 7x the observed worst case.
"""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger
from ai_trading.monitoring.alerts import emit_runtime_alert

logger = get_logger(__name__)

DECISION_LOOP_STALL_THRESHOLD_SECONDS = 900.0
DECISION_LOOP_STALL_REPEAT_SECONDS = 1800.0
DECISION_LOOP_WATCHDOG_CHECK_SECONDS = 30.0

STALL_EVENT = "ALERT_DECISION_LOOP_STALLED"
RECOVERED_EVENT = "DECISION_LOOP_RECOVERED"
HEALTH_ATTENTION_FLAG = "decision_loop_stalled"

_ET = ZoneInfo("America/New_York")


def _as_utc(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _fresh_progress(armed_at: datetime | None, last_progress_at: datetime | None) -> datetime | None:
    """Progress recorded since arming; progress from before arming is not a baseline."""

    if armed_at is None or last_progress_at is None or last_progress_at < armed_at:
        return None
    return last_progress_at


class _DecisionLoopWatchdog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._armed_at: datetime | None = None
        self._last_progress_at: datetime | None = None
        self._last_cycle_index: int | None = None
        self._stalled = False
        self._stall_baseline: datetime | None = None
        self._last_alert_at: datetime | None = None
        self._alerts_sent = 0
        self._session_cache: tuple[date, tuple[datetime, datetime] | None] | None = None
        self._calendar_error_day: date | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def arm(self, *, now: datetime | None = None) -> None:
        with self._lock:
            self._armed_at = _as_utc(now)
            self._stalled = False
            self._stall_baseline = None
            self._last_alert_at = None
            self._alerts_sent = 0

    def disarm(self) -> None:
        with self._lock:
            self._armed_at = None
            self._stalled = False
            self._stall_baseline = None

    def note_progress(self, *, cycle_index: int | None = None, now: datetime | None = None) -> None:
        stamp = _as_utc(now)
        with self._lock:
            self._last_progress_at = stamp
            if cycle_index is not None:
                self._last_cycle_index = int(cycle_index)

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

    def evaluate(self, *, now: datetime | None = None) -> dict[str, Any] | None:
        """Check once; return the stall alert payload when one was emitted."""

        now_utc = _as_utc(now)
        with self._lock:
            armed_at = self._armed_at
            if armed_at is None:
                return None
            last_progress_at = _fresh_progress(armed_at, self._last_progress_at)
            baseline = last_progress_at or armed_at
            stall_baseline = self._stall_baseline
            if self._stalled and stall_baseline is not None and baseline > stall_baseline:
                recovered = {
                    "stalled_seconds": round((baseline - stall_baseline).total_seconds(), 1),
                    "last_cycle_index": self._last_cycle_index,
                    "last_progress_ts": _iso(baseline),
                    "alerts_sent": self._alerts_sent,
                }
                self._stalled = False
                self._stall_baseline = None
                self._last_alert_at = None
                self._alerts_sent = 0
            else:
                recovered = None
        if recovered is not None:
            logger.warning(RECOVERED_EVENT, extra=recovered)
            return None

        age_seconds = max(0.0, (now_utc - baseline).total_seconds())
        if age_seconds <= DECISION_LOOP_STALL_THRESHOLD_SECONDS:
            return None
        if not self._in_regular_session(now_utc):
            return None

        with self._lock:
            if not self._stalled:
                self._stalled = True
                self._stall_baseline = baseline
            last_alert_at = self._last_alert_at
            due = (
                last_alert_at is None
                or (now_utc - last_alert_at).total_seconds() >= DECISION_LOOP_STALL_REPEAT_SECONDS
            )
            if not due:
                return None
            self._last_alert_at = now_utc
            self._alerts_sent += 1
            details: dict[str, Any] = {
                "seconds_since_progress": round(age_seconds, 1),
                "threshold_seconds": DECISION_LOOP_STALL_THRESHOLD_SECONDS,
                "last_cycle_index": self._last_cycle_index,
                "last_progress_ts": _iso(last_progress_at),
                "reason": "no_cycle_since_start" if last_progress_at is None else "no_cycle_progress",
                "alert_number": self._alerts_sent,
                "repeat_seconds": DECISION_LOOP_STALL_REPEAT_SECONDS,
                "alert_only": True,
            }
        emit_runtime_alert(STALL_EVENT, severity="error", details=details)
        return details

    def snapshot(self, *, now: datetime | None = None) -> dict[str, Any]:
        now_utc = _as_utc(now)
        with self._lock:
            armed_at = self._armed_at
            last_progress_at = self._last_progress_at
            baseline = _fresh_progress(armed_at, last_progress_at) or armed_at
            age_seconds = (
                round(max(0.0, (now_utc - baseline).total_seconds()), 1)
                if baseline is not None
                else None
            )
            return {
                "monitored": armed_at is not None,
                "stalled": bool(self._stalled) and armed_at is not None,
                "alert_only": True,
                "last_progress_ts": _iso(last_progress_at),
                "last_cycle_index": self._last_cycle_index,
                "seconds_since_progress": age_seconds,
                "stall_threshold_seconds": DECISION_LOOP_STALL_THRESHOLD_SECONDS,
                "stalled_since_ts": _iso(self._stall_baseline) if self._stalled else None,
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

    def start(self, *, check_seconds: float = DECISION_LOOP_WATCHDOG_CHECK_SECONDS) -> threading.Thread:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
        self.arm()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(stop_event, max(1.0, float(check_seconds))),
            name="decision-loop-watchdog",
            daemon=True,
        )
        with self._lock:
            self._stop_event = stop_event
            self._thread = thread
        thread.start()
        logger.info(
            "DECISION_LOOP_WATCHDOG_STARTED",
            extra={
                "threshold_seconds": DECISION_LOOP_STALL_THRESHOLD_SECONDS,
                "repeat_seconds": DECISION_LOOP_STALL_REPEAT_SECONDS,
                "check_seconds": max(1.0, float(check_seconds)),
            },
        )
        return thread

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            self._stop_event = None
            self._thread = None
        if stop_event is not None:
            stop_event.set()
        self.disarm()


_WATCHDOG = _DecisionLoopWatchdog()


def start_decision_loop_watchdog(
    *, check_seconds: float = DECISION_LOOP_WATCHDOG_CHECK_SECONDS
) -> threading.Thread:
    """Arm the watchdog and start its daemon thread (idempotent)."""

    return _WATCHDOG.start(check_seconds=check_seconds)


def stop_decision_loop_watchdog() -> None:
    """Stop the watchdog thread and disarm it (the loop has exited)."""

    _WATCHDOG.stop()


def note_decision_loop_progress(*, cycle_index: int | None = None, now: datetime | None = None) -> None:
    """Record that the scheduler loop finished an iteration."""

    _WATCHDOG.note_progress(cycle_index=cycle_index, now=now)


def check_decision_loop_stall(*, now: datetime | None = None) -> dict[str, Any] | None:
    """Run one watchdog check; return the alert payload if a stall alert was emitted."""

    return _WATCHDOG.evaluate(now=now)


def get_decision_loop_snapshot(*, now: datetime | None = None) -> dict[str, Any]:
    """Health view of the watchdog; ``stalled`` is the non-fatal attention flag source."""

    return _WATCHDOG.snapshot(now=now)


def _reset_decision_loop_watchdog_for_tests() -> None:
    global _WATCHDOG
    _WATCHDOG.stop()
    _WATCHDOG = _DecisionLoopWatchdog()


__all__ = [
    "DECISION_LOOP_STALL_REPEAT_SECONDS",
    "DECISION_LOOP_STALL_THRESHOLD_SECONDS",
    "DECISION_LOOP_WATCHDOG_CHECK_SECONDS",
    "HEALTH_ATTENTION_FLAG",
    "RECOVERED_EVENT",
    "STALL_EVENT",
    "check_decision_loop_stall",
    "get_decision_loop_snapshot",
    "note_decision_loop_progress",
    "start_decision_loop_watchdog",
    "stop_decision_loop_watchdog",
]
