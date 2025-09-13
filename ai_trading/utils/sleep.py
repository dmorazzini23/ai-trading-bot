from __future__ import annotations

import time as _time

__all__ = ["sleep"]

# Capture the original OS-level sleep to avoid monkeypatch interference
_real_sleep = _time.sleep


def sleep(seconds: float | int) -> float:
    """Sleep for ``seconds`` using :func:`time.sleep` and report elapsed time.

    ``time.sleep`` is captured at import time so monkeypatching ``time.sleep``
    later will not affect this helper. The elapsed duration is measured using
    :func:`time.monotonic` and returned to the caller. If ``seconds`` is zero or
    negative, the function returns ``0.0`` without sleeping.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    if s <= 0:
        return 0.0
    start = _time.monotonic()
    _real_sleep(s)
    return _time.monotonic() - start
