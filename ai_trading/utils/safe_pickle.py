"""Retired helpers for generic pickle-style serialization."""
from __future__ import annotations

def unsafe_model_deserialization_allowed() -> bool:
    """Return whether generic runtime pickle-style loads are allowed."""

    return False


def require_unsafe_model_deserialization(*, scope: str) -> None:
    """Fail closed because generic pickle-style loads are retired."""

    raise RuntimeError(
        f"{scope} uses retired generic model deserialization. "
        "Use JSON-safe inline artifacts or explicit approved runtime model paths instead."
    )


def dumps(obj):
    require_unsafe_model_deserialization(scope="safe_pickle.dumps")
    raise AssertionError("unreachable")


def loads(b):
    require_unsafe_model_deserialization(scope="safe_pickle.loads")
    raise AssertionError("unreachable")


__all__ = [
    "dumps",
    "loads",
    "require_unsafe_model_deserialization",
    "unsafe_model_deserialization_allowed",
]
