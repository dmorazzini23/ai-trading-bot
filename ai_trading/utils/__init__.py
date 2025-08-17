from __future__ import annotations

HTTP_TIMEOUT_S: float = 10.0
SUBPROCESS_TIMEOUT_S: float = 8.0


def clamp_timeout(
    v: float | int | None, *, lo: float = 0.1, hi: float = 60.0, default: float = 10.0
) -> float:
    if v is None:
        return float(default)
    try:
        fv = float(v)
    except Exception:
        return float(default)
    return max(lo, min(hi, fv))


def get_process_manager():
    # Lazy import to honor import-contract
    from . import process_manager  # type: ignore

    return process_manager


__all__ = [
    "HTTP_TIMEOUT_S",
    "SUBPROCESS_TIMEOUT_S",
    "clamp_timeout",
    "get_process_manager",
]
