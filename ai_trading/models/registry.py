from __future__ import annotations

"""Lightweight model path registry.

This module tracks model names and their associated filesystem paths. It
provides helpers to register and retrieve model paths while avoiding silent
conflicts.
"""
from pathlib import Path

try:  # pragma: no cover - executed on first import
    _REGISTRY
except NameError:  # pragma: no cover - executed on first import
    _REGISTRY: dict[str, Path] = {}


def register_model(name: str, path: str | Path) -> Path:
    """Register ``name`` to ``path`` and return the stored path.

    If ``name`` is already registered to the same path, that existing path is
    returned. A ``ValueError`` is raised when attempting to re-register ``name``
    with a different path to avoid accidental divergence.
    """
    p = Path(path).resolve()
    existing = _REGISTRY.get(name)
    if existing is not None:
        if existing == p:
            return existing
        msg = f"Model '{name}' already registered to '{existing}'"
        raise ValueError(msg)
    _REGISTRY[name] = p
    return p


def get_model_path(name: str) -> Path | None:
    """Return the path registered for ``name`` or ``None`` if missing."""
    return _REGISTRY.get(name)


def reset_registry() -> dict[str, Path]:
    """Clear and return the internal registry (for tests)."""
    global _REGISTRY
    _REGISTRY = {}
    return _REGISTRY


def list_models() -> dict[str, Path]:
    """Return a copy of the registry mapping."""
    return dict(_REGISTRY)


__all__ = ["register_model", "get_model_path", "reset_registry", "list_models"]
