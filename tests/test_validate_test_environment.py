from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("scripts/validate_test_environment.py")
    spec = importlib.util.spec_from_file_location("validate_test_environment", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_test_environment_checks_full_suite_packages():
    module = _load_module()
    required = {name for name, _package in module.REQUIRED_PACKAGES}

    assert {
        "sqlalchemy",
        "hypothesis",
        "freezegun",
        "tenacity",
        "psutil",
        "pytest_asyncio",
        "hmmlearn",
        "psycopg",
    } <= required


def test_validate_test_environment_returns_nonzero_for_missing(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "REQUIRED_PACKAGES",
        [("definitely_missing_mod", "definitely-missing-package")],
    )

    assert module.main() == 1


def test_validate_test_environment_returns_zero_for_available(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "REQUIRED_PACKAGES", [("sys", "sys")])

    assert module.main() == 0
