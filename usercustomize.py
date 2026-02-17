"""Extend unittest.mock.patch.dict to keep sys.modules stable during tests."""
from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from typing import Any, cast
import sys
import types
import unittest.mock as _mock
_ORIGINAL_MODULES: dict[str, types.ModuleType | None] = dict(sys.modules)
_ORIGINAL_CLEAR_DICT = getattr(_mock, "_clear_dict", None)


def _restore_modules() -> None:
    sys.modules.update({name: module for name, module in _ORIGINAL_MODULES.items() if module is not None})


class _SafePatchDict(_mock._patch_dict):  # pragma: no cover - behavior exercised indirectly in tests
    def __enter__(self) -> MutableMapping[object, object]:
        result = cast(MutableMapping[object, object], super().__enter__())
        self._maybe_restore_modules()
        return result

    def __exit__(self, *args: object) -> bool:
        exit_result = False
        try:
            exit_result = cast(bool, super().__exit__(*args))
        finally:
            self._maybe_restore_modules()
        return exit_result

    def _maybe_restore_modules(self) -> None:
        if self.clear and self.in_dict is sys.modules:
            _restore_modules()


if hasattr(_mock.patch, "dict"):
    _mock.patch.dict = _SafePatchDict
setattr(_mock, "_patch_dict", _SafePatchDict)


def _ensure_import_machinery(modules: dict[str, types.ModuleType]) -> None:
    for name in ("logging.config", "logging.handlers"):
        if name in modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            continue


def _safe_clear_dict(  # pragma: no cover - behavior exercised in tests
    target: MutableMapping[object, object], *args: object, **kwargs: object
) -> Any:
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
                _ensure_import_machinery(cast(dict[str, types.ModuleType], target))
        return result
    target.clear()
    if preserved is not None:
        target.update(preserved)
        _ensure_import_machinery(cast(dict[str, types.ModuleType], target))
    return None


setattr(_mock, "_clear_dict", _safe_clear_dict)


try:  # pragma: no cover - optional dependency
    import freezegun
except Exception:
    freezegun = None
else:
    _orig_freeze_time: Any = freezegun.freeze_time

    def _freeze_time_with_real_asyncio(*args: object, **kwargs: object) -> Any:
        kwargs.setdefault("real_asyncio", True)
        return _orig_freeze_time(*args, **kwargs)

    freezegun.freeze_time = _freeze_time_with_real_asyncio
    if hasattr(freezegun, "api") and hasattr(freezegun.api, "freeze_time"):
        freezegun.api.freeze_time = _freeze_time_with_real_asyncio
