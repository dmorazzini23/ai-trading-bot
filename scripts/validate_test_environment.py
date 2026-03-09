"""Validate the local environment needed for the full test suite."""

from __future__ import annotations

import importlib
import sys

FULL_TEST_INSTALL_HINT = "make install-dev"

REQUIRED_PACKAGES = [
    ("pytest", "pytest"),
    ("pytest_asyncio", "pytest-asyncio"),
    ("hypothesis", "hypothesis"),
    ("freezegun", "freezegun"),
    ("sqlalchemy", "SQLAlchemy"),
    ("psycopg", "psycopg[binary]"),
    ("tenacity", "tenacity"),
    ("psutil", "psutil"),
    ("hmmlearn", "hmmlearn"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
]


def check_package(import_name: str, package_name: str) -> tuple[bool, str]:
    """Check whether *import_name* is importable."""

    try:
        module = importlib.import_module(import_name)
    except ImportError as exc:
        return False, str(exc)
    return True, getattr(module, "__version__", "unknown")


def main() -> int:
    """Return 0 when the full test environment is available."""

    missing_packages: list[tuple[str, str]] = []

    for import_name, package_name in REQUIRED_PACKAGES:
        success, info = check_package(import_name, package_name)
        if not success:
            missing_packages.append((package_name, info))

    if not missing_packages:
        print("FULL_TEST_ENVIRONMENT_OK")
        return 0

    print("FULL_TEST_ENVIRONMENT_MISSING")
    for package_name, error in missing_packages:
        print(f"- {package_name}: {error}")
    print(f"Install with: {FULL_TEST_INSTALL_HINT}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
