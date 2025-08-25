"""Optional dependency preflight checks (opt-in).

Call ``assert_core_dependencies`` to verify that expected runtime
packages are available before deeper imports occur.
"""

from __future__ import annotations

import importlib

_CORE = [
    "numpy",
    "pandas",
    "pydantic",
    "pytz",
]


def assert_core_dependencies() -> None:
    missing: list[str] = []
    for mod in _CORE:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        raise RuntimeError(
            "Missing core Python packages: " + ", ".join(missing)
            + "\nInstall with: pip install -r requirements.txt"
        )

