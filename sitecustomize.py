"""Testing helper to keep sys.modules intact during patch.dict(clear=True)."""
from __future__ import annotations

import importlib
import sys
import types
import unittest.mock as _mock

_ORIGINAL_PATCH_DICT = _mock.patch.dict
_ORIGINAL_CLEAR_DICT = getattr(_mock, "_clear_dict", None)
_ORIGINAL_MODULES: dict[str, types.ModuleType | None] = dict(sys.modules)


def _safe_patch_dict(in_dict, values=(), clear: bool = False, **kwargs):  # pragma: no cover
    ctx = _ORIGINAL_PATCH_DICT(in_dict, values, clear, **kwargs)
    if clear and in_dict is sys.modules:
        original_enter = ctx.__enter__
        original_exit = ctx.__exit__

        def _enter() -> object:
            result = original_enter()
            sys.modules.update({name: module for name, module in _ORIGINAL_MODULES.items() if module is not None})
            return result

        def _exit(exc_type, exc_val, exc_tb) -> bool:
            try:
                return original_exit(exc_type, exc_val, exc_tb)
            finally:
                sys.modules.update({name: module for name, module in _ORIGINAL_MODULES.items() if module is not None})

        ctx.__enter__ = _enter
        ctx.__exit__ = _exit
    return ctx


_mock.patch.dict = _safe_patch_dict


def _safe_clear_dict(target, *args, **kwargs):  # pragma: no cover - simple wrapper
    preserved: dict[str, types.ModuleType] | None = None
    if target is sys.modules:
        preserved = {
            name: module
            for name, module in _ORIGINAL_MODULES.items()
            if module is not None
        }
    if callable(_ORIGINAL_CLEAR_DICT):
        try:
            result = _ORIGINAL_CLEAR_DICT(target, *args, **kwargs)
        finally:
            if preserved is not None:
                target.update(preserved)
                _ensure_import_machinery(target)
        return result
    target.clear()
    if preserved is not None:
        target.update(preserved)
        _ensure_import_machinery(target)
    return None


_mock._clear_dict = _safe_clear_dict


def _ensure_import_machinery(modules: dict[str, types.ModuleType]) -> None:
    for name in ("logging.config", "logging.handlers"):
        if name not in modules:
            try:
                importlib.import_module(name)
            except Exception:  # pragma: no cover - best effort
                continue
