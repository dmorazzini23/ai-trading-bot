from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


FLAG = "AI_TRADING_IMPORT_SANITY"


def _flag_enabled() -> bool:
    return os.getenv(FLAG) == "1"


@pytest.mark.skipif(not _flag_enabled(), reason="import sanity guard disabled")
def test_python_dotenv_is_resolved(_monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("dotenv", None)
    module = importlib.import_module("dotenv")
    assert hasattr(module, "dotenv_values"), "python-dotenv should expose dotenv_values"
    module_path = Path(getattr(module, "__file__", ""))
    print(str(module_path))
    assert module_path.exists(), "expected python-dotenv module path to exist"
    assert any(part in {"site-packages", "dist-packages"} for part in module_path.parts)
    from dotenv import dotenv_values

    assert callable(dotenv_values)
