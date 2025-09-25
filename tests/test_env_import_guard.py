from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trading.util import env_check


def test_guard_passes_with_python_dotenv() -> None:
    env_check.assert_dotenv_not_shadowed()


def test_guard_detects_shadowed_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(env_check.__file__).resolve().parents[2]
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "dotenv":
            return SimpleNamespace(origin=str(repo_root / "dotenv/__init__.py"))
        return real_find_spec(name)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(env_check.DotenvImportError) as exc:
        env_check.assert_dotenv_not_shadowed()

    assert "python-dotenv is shadowed" in str(exc.value)


def test_guard_allows_repo_local_virtualenv(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(env_check.__file__).resolve().parents[2]
    real_find_spec = importlib.util.find_spec

    venv_origin = repo_root / "venv/lib/python3.12/site-packages/dotenv/__init__.py"

    def fake_find_spec(name: str):
        if name == "dotenv":
            return SimpleNamespace(origin=str(venv_origin))
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    # Should not raise when python-dotenv lives inside an in-repo virtualenv
    # site-packages directory.
    env_check.assert_dotenv_not_shadowed()
