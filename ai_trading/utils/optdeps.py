from __future__ import annotations

"""Lightweight helpers for optional dependencies."""

from types import ModuleType
from typing import Any, Mapping, Optional

__all__ = ["module_ok", "OptionalDependencyError"]


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


def module_ok(mod: Optional[ModuleType | Any]) -> bool:
    """Return True if module/object is not None."""
    return mod is not None

