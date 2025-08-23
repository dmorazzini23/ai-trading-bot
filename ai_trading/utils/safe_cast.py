from __future__ import annotations

def as_int(x, default: int=0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def as_float(x, default: float=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default