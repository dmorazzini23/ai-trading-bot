from __future__ import annotations

import time
from contextlib import contextmanager

@contextmanager
def StageTimer(logger, stage_name: str, **extra):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        try:
            logger.info('STAGE_TIMING', extra={'stage': stage_name, 'elapsed_ms': dt_ms, **extra})
        except (KeyError, ValueError, TypeError):
            pass

class SoftBudget:

    def __init__(self, millis: int):
        self.budget_ms = max(0, int(millis))
        self._start_ns: int | None = None

    def __enter__(self) -> "SoftBudget":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def reset(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def _ensure_started(self) -> int:
        if self._start_ns is None:
            self.reset()
        assert self._start_ns is not None  # for type checkers
        return self._start_ns

    def elapsed_ms(self) -> int:
        start = self._ensure_started()
        elapsed_ns = max(time.perf_counter_ns() - start, 0)
        elapsed_ms = (elapsed_ns + 999_999) // 1_000_000
        return max(1, int(elapsed_ms))

    def over_budget(self) -> bool:
        return self.elapsed_ms() >= self.budget_ms

    def remaining(self) -> float:
        start = self._ensure_started()
        elapsed_ns = max(time.perf_counter_ns() - start, 0)
        remaining_ns = (self.budget_ms * 1_000_000) - elapsed_ns
        if remaining_ns <= 0:
            return 0.0
        remaining_ms = (remaining_ns + 999_999) // 1_000_000
        return int(remaining_ms) / 1000.0

    def over(self) -> bool:  # Backward compatibility
        return self.over_budget()
