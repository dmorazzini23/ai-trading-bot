"""Test-only import guards ensuring python-dotenv is the resolved module."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Iterable

_GUARD_FLAG = "AI_TRADING_IMPORT_SANITY"


def _iter_site_package_paths() -> Iterable[str]:
    """Yield candidate paths that typically host third-party packages."""

    for path in sys.path:
        if not isinstance(path, str):
            continue
        lowered = path.lower()
        if "site-packages" in lowered or "dist-packages" in lowered:
            yield path
    try:
        import sysconfig
    except Exception:  # pragma: no cover - extremely unlikely
        return
    for key in ("purelib", "platlib"):
        candidate = sysconfig.get_path(key)
        if candidate:
            yield candidate


def _load_canonical_dotenv() -> ModuleType:
    """Load the python-dotenv implementation even when shadowed."""

    module = importlib.import_module("dotenv")
    if hasattr(module, "dotenv_values"):
        return module

    for base in _iter_site_package_paths():
        module_path = os.path.join(base, "dotenv", "__init__.py")
        if not os.path.exists(module_path):
            continue
        spec = importlib.util.spec_from_file_location("dotenv", module_path)
        if spec and spec.loader:
            canonical = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(canonical)  # type: ignore[misc]
            if hasattr(canonical, "dotenv_values"):
                return canonical

    path_hint = getattr(module, "__file__", "unknown")
    raise ImportError(
        "python-dotenv import sanity failed: expected `dotenv_values` on module "
        f"loaded from {path_hint}. Ensure python-dotenv is installed and no test "
        "stubs shadow the package."
    )


if os.getenv(_GUARD_FLAG) == "1":
    canonical_module = _load_canonical_dotenv()
    sys.modules["dotenv"] = canonical_module
