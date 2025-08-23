from __future__ import annotations

import time

# Default HTTP timeout (seconds)
HTTP_TIMEOUT: float = 10.0


def clamp_timeout(value: float | None) -> float:
    """Return a sane timeout (value if positive else default)."""
    try:
        if value is not None and float(value) > 0:
            return float(value)
    except Exception:
        pass
    return HTTP_TIMEOUT


def sleep(seconds: float) -> None:
    """Thin wrapper for testability / monkeypatching."""
    time.sleep(max(0.0, float(seconds)))
