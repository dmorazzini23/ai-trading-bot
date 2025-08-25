from __future__ import annotations

"""Lightweight helpers for optional dependencies."""

from importlib import import_module
from types import ModuleType
from typing import Any, Optional
from dataclasses import dataclass

__all__ = ["optional_import", "module_ok", "OptionalDependencyError"]


# AI-AGENT-REF: centralized optional dependency error
@dataclass
class OptionalDependencyError(ImportError):
    """Clear error for missing optional packages."""

    name: str
    purpose: Optional[str] = None
    extra: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        bits = [f"Missing optional dependency: {self.name}."]
        if self.purpose:
            bits.append(f"Needed for: {self.purpose}.")
        if self.extra:
            bits.append(f"Install with: {self.extra}")
        else:
            bits.append(f"Try: pip install {self.name}")
        return " ".join(bits)


def optional_import(
    name: str,
    *,
    required: bool = False,
    attr: Optional[str] = None,
    purpose: Optional[str] = None,
    extra: Optional[str] = None,
) -> Any | None:
    """Import a module (and optionally attribute) if available."""

    try:
        mod = import_module(name)
    except Exception as e:  # ImportError / ModuleNotFoundError
        if required:
            raise OptionalDependencyError(name=name, purpose=purpose, extra=extra) from e
        return None

    if attr:
        try:
            return getattr(mod, attr)
        except AttributeError as e:
            if required:
                raise OptionalDependencyError(name=name, purpose=purpose, extra=extra) from e
            return None
    return mod


def module_ok(mod: Optional[ModuleType | Any]) -> bool:
    """Return True if module/object is not None."""
    return mod is not None
