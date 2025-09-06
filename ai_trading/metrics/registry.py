from __future__ import annotations

from ai_trading.metrics import CollectorRegistry

try:  # pragma: no cover - executed on first import
    _REGISTRY
except NameError:  # pragma: no cover - executed on first import
    _REGISTRY = CollectorRegistry()


def get_registry() -> CollectorRegistry:
    """Return the module-level :class:`CollectorRegistry` instance."""
    return _REGISTRY


def reset_registry(registry: CollectorRegistry | None = None) -> CollectorRegistry:
    """Replace and return the active :class:`CollectorRegistry`.

    ``registry`` defaults to a new :class:`CollectorRegistry` when ``None``.
    """
    global _REGISTRY
    _REGISTRY = registry or CollectorRegistry()
    return _REGISTRY


__all__ = ["get_registry", "reset_registry"]
