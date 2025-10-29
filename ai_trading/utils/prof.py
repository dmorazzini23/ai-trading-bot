from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable

_TIMING_LEVEL_CACHE: tuple[str | None, str | None, int | None] | None = None


def _resolve_timing_level() -> int | None:
    """Return the configured log level for stage timing events."""

    global _TIMING_LEVEL_CACHE
    primary = os.getenv("AI_TRADING_LOG_TIMINGS_LEVEL")
    fallback = os.getenv("LOG_TIMINGS_LEVEL")
    if (
        _TIMING_LEVEL_CACHE is not None
        and _TIMING_LEVEL_CACHE[0] == primary
        and _TIMING_LEVEL_CACHE[1] == fallback
    ):
        return _TIMING_LEVEL_CACHE[2]

    raw_level = primary if primary is not None else fallback
    if raw_level is None:
        level: int | None = logging.DEBUG
    else:
        value = str(raw_level).strip().upper()
        if value in {"OFF", "NONE", "DISABLED"}:
            level = None
        else:
            level = getattr(logging, value, logging.DEBUG)
    _TIMING_LEVEL_CACHE = (primary, fallback, level)
    return level


def _log_at_level(logger: Any, level: int, message: str, *, extra: dict[str, Any]) -> None:
    """Emit ``message`` at ``level`` while honouring common logger helpers."""

    if level == logging.DEBUG and hasattr(logger, "debug"):
        logger.debug(message, extra=extra)
    elif level == logging.INFO and hasattr(logger, "info"):
        logger.info(message, extra=extra)
    elif level == logging.WARNING and hasattr(logger, "warning"):
        logger.warning(message, extra=extra)
    elif level == logging.ERROR and hasattr(logger, "error"):
        logger.error(message, extra=extra)
    else:
        logger.log(level, message, extra=extra)


@contextmanager
def StageTimer(logger: Any, stage_name: str, *, override_ms: float | None = None, **extra: Any) -> None:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        level = _resolve_timing_level()
        if level is None:
            return
        dt_ms = override_ms
        if dt_ms is None:
            dt_ms = int((time.perf_counter() - t0) * 1000)
        else:
            try:
                dt_ms = int(max(0.0, float(dt_ms)))
            except (TypeError, ValueError):
                dt_ms = int((time.perf_counter() - t0) * 1000)
        payload = {"stage": stage_name, "elapsed_ms": dt_ms, **extra}
        try:
            if logger.isEnabledFor(level):
                _log_at_level(logger, level, "STAGE_TIMING", extra=payload)
        except (KeyError, ValueError, TypeError):
            pass

class SoftBudget:
    """Track a soft millisecond budget using monotonic time."""

    def __init__(self, millis: int | float):
        self.ms = max(0.0, float(millis))
        self.start_ns = time.perf_counter_ns()
        self.start = self.start_ns / 1_000_000_000  # legacy attribute

    def __enter__(self) -> "SoftBudget":
        self.start_ns = time.perf_counter_ns()
        self.start = self.start_ns / 1_000_000_000
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds since the most recent reset."""

        elapsed_ns = time.perf_counter_ns() - self.start_ns
        if elapsed_ns < 0:
            elapsed_ns = 0
        whole_ms = elapsed_ns // 1_000_000
        if elapsed_ns > 0 and whole_ms == 0:
            return 1.0
        return float(whole_ms)

    def remaining(self) -> float:
        """Return milliseconds remaining before the budget is exceeded."""

        remaining_ms = self.ms - self.elapsed_ms()
        return remaining_ms if remaining_ms > 0.0 else 0.0

    def over_budget(self) -> bool:
        """Return True if the elapsed time has exceeded the budget."""

        return self.remaining() <= 0.0

    def over(self) -> bool:
        """Backward compatibility alias for :meth:`over_budget`."""

        return self.over_budget()
