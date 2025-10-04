"""User-level interpreter bootstrap for tests.

Ensures ``sys.modules`` retains core stdlib modules even when cleared via
``patch.dict``.  This avoids recursion inside pytest's assertion rewrite hook
when importing our package after ``sys.modules`` was emptied.
"""
from __future__ import annotations

import sys

_ESSENTIALS: dict[str, object | None] = {}
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
    except Exception:  # pragma: no cover - optional modules may not be available yet
        module = None
    _ESSENTIALS[name] = module


class _ProtectedModules(dict):
    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        for key, module in _ESSENTIALS.items():
            if module is not None:
                super().__setitem__(key, module)


if not isinstance(sys.modules, _ProtectedModules):
    sys.modules = _ProtectedModules(sys.modules)
