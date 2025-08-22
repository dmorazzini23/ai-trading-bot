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
            logger.info("STAGE_TIMING", extra={"stage": stage_name, "elapsed_ms": dt_ms, **extra})
        # noqa: BLE001 TODO: narrow exception
        except Exception:
            pass


class SoftBudget:
    def __init__(self, interval_sec: float, fraction: float):
        self._deadline = time.monotonic() + max(0.0, interval_sec) * max(0.1, min(1.0, fraction))
        self._start = time.monotonic()

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)

    def over(self) -> bool:
        return time.monotonic() >= self._deadline
