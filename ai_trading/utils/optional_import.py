from __future__ import annotations

from importlib import import_module
from typing import Any


def optional_import(module: str, attr: str | None = None) -> Any | None:
    """Import a module or attribute if available, otherwise return ``None``."""
    try:
        mod = import_module(module)
    except ModuleNotFoundError:
        return None
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except AttributeError:
        return None
