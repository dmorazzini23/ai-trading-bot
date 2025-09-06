from __future__ import annotations

import time as _time
from time import perf_counter as _perf

__all__ = ["sleep"]

# Capture the original OS-level sleep to avoid monkeypatch interference
_real_sleep = _time.sleep


def sleep(seconds: float | int) -> float:
    """Sleep for at least ``seconds`` seconds and return actual duration.

    The underlying :func:`time.sleep` function is captured at import time so
    later monkeypatching of ``time.sleep`` will not short-circuit this helper.
    A minimum delay of ~10ms is enforced to ensure a measurable pause even when
    ``seconds`` is 0 or negative.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = 0.0
    target = max(s, 0.01)
    start = _perf()
    _real_sleep(target)
    elapsed = _perf() - start
    tries = 0
    while elapsed < 0.009 and tries < 5:
        _real_sleep(0.005)
        elapsed = _perf() - start
        tries += 1
    return elapsed
