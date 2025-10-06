"""Extend unittest.mock.patch.dict to keep sys.modules stable during tests."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast
import sys
import types
import unittest.mock as _mock
_ORIGINAL_MODULES: dict[str, types.ModuleType | None] = dict(sys.modules)


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
