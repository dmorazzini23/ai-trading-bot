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
def StageTimer(logger: Any, stage_name: str, **extra: Any) -> None:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        level = _resolve_timing_level()
        if level is None:
            return
        dt_ms = int((time.perf_counter() - t0) * 1000)
        payload = {"stage": stage_name, "elapsed_ms": dt_ms, **extra}
        try:
            if logger.isEnabledFor(level):
                _log_at_level(logger, level, "STAGE_TIMING", extra=payload)
        except (KeyError, ValueError, TypeError):
            pass

class SoftBudget:

    def __init__(self, millis: int):
        self.budget_ms = max(0, int(millis))
        self._start_ns: int | None = time.perf_counter_ns()
        self._last_elapsed_ns: int = 0
        self._fractional_ns: int = 0
        self._reported_ms: int = 0

    def __enter__(self) -> "SoftBudget":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def reset(self) -> None:
        self._start_ns = time.perf_counter_ns()
        self._last_elapsed_ns = 0
        self._fractional_ns = 0
        self._reported_ms = 0

    def _elapsed_ns(self) -> int:
        now = time.perf_counter_ns()
        if self._start_ns is None:
            self._start_ns = now
            self._last_elapsed_ns = 0
            self._fractional_ns = 0
            self._reported_ms = 0
            return 0
        elapsed_ns = now - self._start_ns
        return elapsed_ns if elapsed_ns >= 0 else 0

    def elapsed_ms(self) -> int:
        """Return elapsed milliseconds since the most recent reset."""

        elapsed_ns = self._elapsed_ns()
        delta_ns = elapsed_ns - self._last_elapsed_ns
        if delta_ns <= 0:
            return max(1, self._reported_ms)

        self._last_elapsed_ns = elapsed_ns
        self._fractional_ns += delta_ns

        increment, self._fractional_ns = divmod(self._fractional_ns, 1_000_000)
        if increment:
            self._reported_ms += increment

        if self._reported_ms == 0 and self._fractional_ns > 0:
            # Surface a minimal positive tick so extremely short durations
            # are observable without skewing subsequent accumulation.
            return 1

        return self._reported_ms

    def over_budget(self) -> bool:
        return self._elapsed_ns() >= (self.budget_ms * 1_000_000)

    def remaining(self) -> float:
        remaining_ns = (self.budget_ms * 1_000_000) - self._elapsed_ns()
        return 0.0 if remaining_ns <= 0 else round(remaining_ns / 1_000_000_000, 3)

    def over(self) -> bool:  # Backward compatibility
        return self.over_budget()
