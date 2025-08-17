"""Lightweight utility exports with lazy submodule access.

This module intentionally keeps imports minimal to avoid heavy import-time side
effects.  Most helpers live in submodules such as :mod:`ai_trading.utils.base`
or :mod:`ai_trading.utils.determinism` and are loaded on demand using
``__getattr__``.  Only a couple of timeout constants and ``clamp_timeout`` are
eagerly defined here.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - hints only
    from . import process_manager as _process_manager  # noqa: F401

HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "10"))
SUBPROCESS_TIMEOUT_S: float = float(os.getenv("SUBPROCESS_TIMEOUT_S", "5"))


def clamp_timeout(
    t: float | None,
    *,
    default: float = HTTP_TIMEOUT,
    low: float = 0.1,
    high: float = 60.0,
) -> float:
    """Clamp ``t`` to a safe timeout value."""

    if t is None:
        return default
    return max(low, min(float(t), high))


__all__ = ["HTTP_TIMEOUT", "SUBPROCESS_TIMEOUT_S", "clamp_timeout", "get_process_manager"]

# Submodules imported lazily via ``__getattr__`` to preserve the old API while
# keeping this module lightweight.  Only names listed here are exposed as
# modules when accessed via ``from ai_trading.utils import <name>``.
_LAZY_SUBMODULES = {
    "process_manager",
    "http",
    "paths",
    "workers",
    "memory_optimizer",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin passthrough
    if name in _LAZY_SUBMODULES:
        return import_module(f"{__name__}.{name}")
    # Fallback to attributes from ``base`` and ``determinism`` for backwards
    # compatibility.  These imports happen lazily to keep import time minimal.
    for mod_name in ("base", "determinism", "timing"):
        try:
            mod = import_module(f"{__name__}.{mod_name}")
            if hasattr(mod, name):
                return getattr(mod, name)
        except Exception:  # pragma: no cover - optional dependency
            continue
    raise AttributeError(name) from None


def get_process_manager():  # pragma: no cover - thin wrapper
    """Return the lazily imported :mod:`process_manager` module."""

    from . import process_manager  # noqa: WPS433 (allowed lazy import)

    return process_manager


def __dir__() -> list[str]:  # pragma: no cover - simple namespace helper
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))
