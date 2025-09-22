from __future__ import annotations

import math
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
        self._start = time.perf_counter()

    def __enter__(self) -> "SoftBudget":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def reset(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> int:
        elapsed = time.perf_counter() - self._start
        if elapsed <= 0:
            return 0
        return math.ceil(elapsed * 1000)

    def over_budget(self) -> bool:
        return self.elapsed_ms() >= self.budget_ms

    def remaining(self) -> float:
        elapsed = time.perf_counter() - self._start
        remaining_ms = self.budget_ms - (elapsed * 1000)
        if remaining_ms <= 0:
            return 0.0
        return math.ceil(remaining_ms) / 1000.0

    def over(self) -> bool:  # Backward compatibility
        return self.over_budget()
