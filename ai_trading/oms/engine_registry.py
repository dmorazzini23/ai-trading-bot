"""Shared SQLAlchemy engine registry for OMS Postgres backends."""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, TypeVar

from ai_trading.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_ENGINE_REGISTRY: dict[str, Any] = {}
_ENGINE_REGISTRY_LOCK = RLock()


def resolve_shared_engine(
    *,
    registry_key: str | None,
    factory: Callable[[], T],
) -> tuple[T, bool]:
    """Return a shared engine for *registry_key* when provided.

    Returns ``(engine, owns_engine)`` where ``owns_engine`` is ``False`` for
    shared registry entries and ``True`` for one-off engines.
    """

    shared_key = str(registry_key or "").strip()
    if not shared_key:
        return factory(), True

    with _ENGINE_REGISTRY_LOCK:
        cached = _ENGINE_REGISTRY.get(shared_key)
        if cached is not None:
            return cached, False
        engine = factory()
        _ENGINE_REGISTRY[shared_key] = engine
        return engine, False


def reset_shared_engines() -> None:
    """Dispose and clear shared engines.

    This is primarily intended for tests that need a clean registry.
    """

    with _ENGINE_REGISTRY_LOCK:
        cached = list(_ENGINE_REGISTRY.values())
        _ENGINE_REGISTRY.clear()
    for engine in cached:
        try:
            dispose = getattr(engine, "dispose", None)
            if callable(dispose):
                dispose()
        except Exception:
            logger.debug("OMS_SHARED_ENGINE_DISPOSE_FAILED", exc_info=True)
