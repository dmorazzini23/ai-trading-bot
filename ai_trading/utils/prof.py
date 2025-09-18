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

    def __init__(self, interval_sec: float, fraction: float):
        now = time.monotonic()
        self._deadline = now + max(0.0, interval_sec) * max(0.1, min(1.0, fraction))
        self._start = now

    def __enter__(self) -> "SoftBudget":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # nothing to clean up and don't suppress exceptions
        return None

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def elapsed_ms(self) -> int:
        elapsed = time.monotonic() - self._start
        if elapsed < 1e-9:
            return 0
        return max(1, math.ceil(elapsed * 1000))

    def over(self) -> bool:
        return time.monotonic() >= self._deadline
