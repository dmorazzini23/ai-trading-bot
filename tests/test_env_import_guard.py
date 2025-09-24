from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from ai_trading.util import env_check


def test_guard_passes_with_python_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_IMPORT_GUARD", "true")
    # Ensure any cached import is cleared so guard re-imports the real module.
    sys.modules.pop("dotenv", None)
    env_check.guard_python_dotenv(force=True)


class _DummyModule(ModuleType):
    pass


def test_guard_detects_shadowed_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_IMPORT_GUARD", "true")
    shadow = _DummyModule("dotenv")
    sys.modules["dotenv"] = shadow
    with pytest.raises(env_check.DotenvImportError) as exc:
        env_check.guard_python_dotenv(force=True)
    assert "python-dotenv not available" in str(exc.value)
    sys.modules.pop("dotenv", None)
    importlib.invalidate_caches()
