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


def register(metric) -> object:
    """Register ``metric`` in the global registry if not already present.

    ``prometheus_client`` will raise ``ValueError`` when attempting to register
    a metric with a name that already exists.  Older versions expose a
    ``_names_to_collectors`` mapping that we can query directly.  This helper
    checks for an existing collector with the same name and returns it instead
    of re-registering, preventing duplicate metrics and avoiding exceptions.
    """

    registry = _REGISTRY
    name = getattr(metric, "_name", getattr(metric, "name", None))
    existing = getattr(registry, "_names_to_collectors", {}).get(name) if name else None
    if existing is not None:
        return existing
    registry.register(metric)
    return metric


__all__ = ["get_registry", "reset_registry", "register"]
