from __future__ import annotations

from importlib import import_module
from typing import Any


def optional_import(module: str, attr: str | None = None) -> Any | None:
    """
    Attempt to import a vendor module (and optionally an attribute).
    Returns the module/attr if importable; otherwise returns None.
    Never raises ImportError; callers must branch on None.
    """
    try:
        mod = import_module(module)
    except Exception:  # noqa: BLE001 - intentional guard at the boundary
        return None
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except Exception:  # attribute may not exist on older versions
        return None
