from __future__ import annotations

import logging
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Callable

_TIMING_LEVEL_CACHE: tuple[str | None, str | None, int | None] | None = None


def _monotonic_ns() -> int:
    """Return a monotonic clock reading in nanoseconds."""

    perf_counter_ns = getattr(time, "perf_counter_ns", None)
    if callable(perf_counter_ns):
        try:
            return perf_counter_ns()
        except (OSError, ValueError):
            pass
    monotonic = getattr(time, "monotonic", None)
    if callable(monotonic):
        try:
            return int(monotonic() * 1_000_000_000)
        except (OSError, ValueError):
            pass
    monotonic_ns = getattr(time, "monotonic_ns", None)
    if callable(monotonic_ns):
        try:
            return monotonic_ns()
        except (OSError, ValueError):
            pass
    return int(time.perf_counter() * 1_000_000_000)


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
        raw_ms: float
        if override_ms is None:
            raw_ms = (time.perf_counter() - t0) * 1000.0
        else:
            try:
                raw_ms = float(override_ms)
            except (TypeError, ValueError):
                raw_ms = (time.perf_counter() - t0) * 1000.0
        if not math.isfinite(raw_ms) or raw_ms < 0.0:
            raw_ms = 0.0
        dt_ms_int = 0
        if raw_ms > 0.0:
            dt_ms_int = 1 if raw_ms < 1.0 else int(math.ceil(raw_ms))
        payload = {"stage": stage_name, "elapsed_ms": dt_ms_int, **extra}
        try:
            if logger.isEnabledFor(level):
                _log_at_level(logger, level, "STAGE_TIMING", extra=payload)
        except (KeyError, ValueError, TypeError):
            pass

class SoftBudget:
    """Track a soft millisecond budget using monotonic time."""

    def __init__(self, millis: int | float):
        self.ms = max(0.0, float(millis))
        self.start_ns = 0
        self.start = 0.0  # legacy attribute
        self.reset()

    def __enter__(self) -> "SoftBudget":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def reset(self) -> None:
        """Reset the budget start time."""

        self.start_ns = _monotonic_ns()
        self.start = self.start_ns / 1_000_000_000

    def _elapsed_ns(self) -> int:
        elapsed_ns = _monotonic_ns() - self.start_ns
        if elapsed_ns < 0:
            return 0
        return int(elapsed_ns)

    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds since the most recent reset."""

        elapsed_ns = self._elapsed_ns()
        whole_ms = elapsed_ns // 1_000_000
        if elapsed_ns > 0 and whole_ms == 0:
            return 1.0
        return float(whole_ms)

    def remaining(self) -> float:
        """Return seconds remaining before the budget is exceeded."""

        remaining_ms = self.ms - self.elapsed_ms()
        if remaining_ms <= 0.0:
            return 0.0
        return remaining_ms / 1000.0

    def over_budget(self) -> bool:
        """Return True if the elapsed time has exceeded the budget."""

        return self._elapsed_ns() >= int(self.ms * 1_000_000)

    def over(self) -> bool:
        """Backward compatibility alias for :meth:`over_budget`."""

        return self.over_budget()
