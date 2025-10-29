"""
Execution timing helpers for measuring broker submit spans.

The trading loop records timing metadata for each broker submission so that
`ai_trading.main` can accurately report the time spent in the execution stage.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Mapping

CycleMetadata = Dict[str, Any]

_cycle_total_seconds = 0.0
_cycle_wall_seconds = 0.0
_cycle_lock = threading.Lock()
_on_span_complete: Callable[[float, CycleMetadata], None] | None = None


def reset_cycle(on_span_complete: Callable[[float, CycleMetadata], None] | None = None) -> None:
    """Reset accumulated timing for a new trading cycle."""

    global _cycle_total_seconds, _cycle_wall_seconds, _on_span_complete
    with _cycle_lock:
        _cycle_total_seconds = 0.0
        _cycle_wall_seconds = 0.0
        _on_span_complete = on_span_complete


def _record_span(elapsed: float, meta: CycleMetadata) -> None:
    global _cycle_total_seconds
    clamped = max(0.0, float(elapsed))
    with _cycle_lock:
        _cycle_total_seconds += clamped
        callback = _on_span_complete
    if callback is not None:
        try:
            callback(clamped, meta)
        except Exception:
            # Individual callbacks should never break execution timing.
            pass


def cycle_seconds() -> float:
    """Return total execution seconds accumulated for the active cycle."""

    with _cycle_lock:
        return max(_cycle_total_seconds, _cycle_wall_seconds)


def record_cycle_wall(elapsed: float, metadata: Mapping[str, Any] | None = None) -> None:
    """Record wall-clock execution time for the active cycle."""

    global _cycle_wall_seconds
    clamped = max(0.0, float(elapsed))
    with _cycle_lock:
        if clamped > _cycle_wall_seconds:
            _cycle_wall_seconds = clamped
        callback = _on_span_complete
    if callback is not None and metadata is not None:
        try:
            callback(clamped, dict(metadata))
        except Exception:
            pass


@contextmanager
def execution_span(logger: Any, **extra: Any):
    """Context manager that records execution timing with optional logging."""

    start = time.perf_counter()
    payload: dict[str, Any] = dict(extra)
    if logger is not None:
        try:
            logger.info("EXECUTE_TIMING_START", extra=payload)
        except Exception:
            pass
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        payload_end = dict(extra)
        payload_end["elapsed_ms"] = int(max(0.0, elapsed) * 1000.0)
        _record_span(elapsed, payload_end)
        if logger is not None:
            try:
                logger.info("EXECUTE_TIMING_END", extra=payload_end)
            except Exception:
                pass

