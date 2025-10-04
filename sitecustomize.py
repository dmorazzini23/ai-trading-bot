"""Interpreter bootstrap helpers for ai_trading test environment.

Wrapping ``sys.modules`` ensures that critical standard library modules survive
``patch.dict(sys.modules, clear=True)`` used by our tests.  Without this guard the
pytest assertion rewrite import hook can recurse while re-importing stdlib
modules, causing the bot engine import tests to fail before library code has a
chance to run.
"""
from __future__ import annotations

import sys

# Capture canonical modules that pytest and importlib rely on while rewriting
essentials: dict[str, object | None] = {}
for name in (
    "sys",
    "builtins",
    "types",
    "importlib",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
    "importlib.machinery",
    "pathlib",
    "posixpath",
    "ntpath",
    "os",
    "collections",
    "_pytest.assertion.rewrite",
):
    try:
        module = __import__(name)
    except Exception:  # pragma: no cover - optional modules may be unavailable
        module = None
    essentials[name] = module


class _ProtectedModules(dict):
    """Dictionary wrapper that restores essential modules after ``clear()``."""

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        for key, module in essentials.items():
            if module is not None:
                super().__setitem__(key, module)


if not isinstance(sys.modules, _ProtectedModules):
    sys.modules = _ProtectedModules(sys.modules)
