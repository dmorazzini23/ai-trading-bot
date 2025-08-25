from __future__ import annotations

"""
Lightweight helpers for optional dependencies.

Usage:
    from ai_trading.utils.optdeps import optional_import, module_ok

    pd = optional_import("pandas")                 # -> module or None
    tz = optional_import("zoneinfo", attr="ZoneInfo")
    np = optional_import("numpy", required=True, install_hint="pip install numpy")
"""

from importlib import import_module
from typing import Any, Optional

__all__ = ["optional_import", "module_ok"]


# AI-AGENT-REF: centralize optional dependency handling

def optional_import(
    name: str,
    *,
    required: bool = False,
    attr: Optional[str] = None,
    install_hint: Optional[str] = None,
) -> Any | None:
    """
    Try to import a module (and optionally fetch an attribute). Return the module/attr
    if available, else None unless `required=True`, in which case raise ImportError
    with a concise, actionable message.

    Args:
        name: import path, e.g. "pandas" or "zoneinfo"
        required: if True, raise on failure (with hint)
        attr: optional attribute name to fetch and return from the imported module
        install_hint: optional human-friendly remedy, e.g. "pip install pandas"
    """
    try:
        mod = import_module(name)
    except Exception as e:
        if required:
            hint = f" Install with `{install_hint}`." if install_hint else ""
            raise ImportError(f"Optional dependency '{name}' is required.{hint}") from e
        return None

    if attr:
        try:
            return getattr(mod, attr)
        except AttributeError as e:
            if required:
                raise ImportError(
                    f"Optional dependency '{name}' lacks required attribute '{attr}'."
                ) from e
            return None

    return mod


def module_ok(name: str, *, allow_missing: bool = True) -> bool:
    """Return True if `import name` succeeds. If allow_missing=False, re-raise ImportError."""
    try:
        import_module(name)
        return True
    except Exception:
        if allow_missing:
            return False
        raise
