from __future__ import annotations

"""Lightweight helpers for optional dependencies."""

from importlib import import_module
from types import ModuleType
from typing import Any, Mapping, Optional

__all__ = ["optional_import", "module_ok", "OptionalDependencyError"]


# AI-AGENT-REF: map modules to extras for install hints
_EXTRAS_BY_PKG: Mapping[str, str] = {
    "pandas": "pandas",
    "matplotlib": "plot",
    "sklearn": "ml",
    "torch": "ml",
    "ta": "ta",
    "talib": "ta",
}


# AI-AGENT-REF: extras-aware optional dependency error
class OptionalDependencyError(ImportError):
    def __init__(
        self,
        package: str,
        *,
        extra: Optional[str] = None,
        feature: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Clear error for missing optional packages with install hint."""

        if message:
            msg = message
        else:
            if extra:
                hint = (
                    extra
                    if extra.strip().startswith("pip ")
                    else f'pip install "ai-trading-bot[{extra}]"'
                )
            else:
                hint = f"pip install {package}"
            suffix = f" for {feature}" if feature else ""
            msg = f"Missing optional dependency '{package}'{suffix}. Install with: {hint}"
        super().__init__(msg)
        self.package = package
        self.name = package  # back-compat
        self.extra = extra
        self.feature = feature
        self.purpose = feature  # back-compat


# AI-AGENT-REF: derive extras hints on optional imports
def optional_import(
    name: str,
    *,
    required: bool = False,
    attr: Optional[str] = None,
    message: Optional[str] = None,
    extra: Optional[str] = None,
    feature: Optional[str] = None,
    purpose: Optional[str] = None,
) -> Any | None:
    """Import a module (and optionally attribute) if available."""

    feature = feature or purpose
    try:
        mod = import_module(name)
    except Exception:
        if required:
            resolved_extra = extra or _EXTRAS_BY_PKG.get(name)
            raise OptionalDependencyError(
                name,
                extra=resolved_extra,
                feature=feature,
                message=message,
            )
        return None

    if attr:
        try:
            return getattr(mod, attr)
        except AttributeError:
            if required:
                resolved_extra = extra or _EXTRAS_BY_PKG.get(name)
                raise OptionalDependencyError(
                    name,
                    extra=resolved_extra,
                    feature=feature,
                    message=message,
                )
            return None
    return mod


def module_ok(mod: Optional[ModuleType | Any]) -> bool:
    """Return True if module/object is not None."""
    return mod is not None

