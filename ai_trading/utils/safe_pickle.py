"""Guarded helpers for generic pickle-style serialization."""
from __future__ import annotations

from ai_trading.config.management import get_env, is_test_runtime

try:  # pragma: no cover - optional dependency
    import cloudpickle as _pickle  # type: ignore
except Exception:  # pragma: no cover - fallback
    import pickle as _pickle  # type: ignore


def unsafe_model_deserialization_allowed() -> bool:
    """Return whether generic runtime pickle-style loads are explicitly allowed."""

    return bool(
        is_test_runtime()
        or get_env("AI_TRADING_ALLOW_UNSAFE_MODEL_DESERIALIZATION", False, cast=bool)
    )


def require_unsafe_model_deserialization(*, scope: str) -> None:
    """Fail closed unless explicitly permitted for test or research runtimes."""

    if unsafe_model_deserialization_allowed():
        return
    raise RuntimeError(
        f"{scope} uses unsafe generic model deserialization and is disabled by default. "
        "Set AI_TRADING_ALLOW_UNSAFE_MODEL_DESERIALIZATION=1 only in controlled research or "
        "migration workflows."
    )


def dumps(obj):
    return _pickle.dumps(obj)


def loads(b):
    require_unsafe_model_deserialization(scope="safe_pickle.loads")
    return _pickle.loads(b)


__all__ = [
    "dumps",
    "loads",
    "require_unsafe_model_deserialization",
    "unsafe_model_deserialization_allowed",
]
