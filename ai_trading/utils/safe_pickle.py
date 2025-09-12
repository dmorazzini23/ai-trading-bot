"""Wrapper around pickle using cloudpickle when available."""
from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import cloudpickle as _pickle  # type: ignore
except Exception:  # pragma: no cover - fallback
    import pickle as _pickle  # type: ignore


def dumps(obj):
    return _pickle.dumps(obj)


def loads(b):
    return _pickle.loads(b)


__all__ = ["dumps", "loads"]

