"""Testing support hooks for unittest.mock.patch.dict."""
from __future__ import annotations

import sys
import types
import unittest.mock as _mock

# Snapshot interpreter modules so we can restore them when tests clear
# ``sys.modules`` via ``patch.dict(..., clear=True)``.
_ORIGINAL_MODULES: dict[str, types.ModuleType | None] = dict(sys.modules)
_ORIGINAL_PATCH_DICT = _mock.patch.dict


def _safe_patch_dict(in_dict, values=(), clear: bool = False, **kwargs):  # pragma: no cover
    ctx = _ORIGINAL_PATCH_DICT(in_dict, values, clear, **kwargs)
    if clear and in_dict is sys.modules:
        original_enter = ctx.__enter__
        original_exit = ctx.__exit__

        def _enter():
            result = original_enter()
            sys.modules.update({k: v for k, v in _ORIGINAL_MODULES.items() if v is not None})
            return result

        def _exit(exc_type, exc_val, exc_tb):
            try:
                return original_exit(exc_type, exc_val, exc_tb)
            finally:
                sys.modules.update({k: v for k, v in _ORIGINAL_MODULES.items() if v is not None})

        ctx.__enter__ = _enter
        ctx.__exit__ = _exit
    return ctx


_mock.patch.dict = _safe_patch_dict
