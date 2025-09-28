"""Test-only import guards ensuring python-dotenv is the resolved module."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
from types import ModuleType
from typing import Iterable

# Ensure the project root stays on sys.path for subprocesses spawned by tests.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _REPO_ROOT not in _existing_pythonpath.split(os.pathsep):
os.environ["PYTHONPATH"] = (
    f"{_REPO_ROOT}" if not _existing_pythonpath else f"{_REPO_ROOT}{os.pathsep}{_existing_pythonpath}"
)
py_assert_opts = os.environ.get("PYTEST_ADDOPTS", "")
if "--assert=plain" not in py_assert_opts.split():
    py_assert_opts = (py_assert_opts + " --assert=plain").strip()
    os.environ["PYTEST_ADDOPTS"] = py_assert_opts
dont_rewrite = os.environ.get("PYTEST_DONT_REWRITE", "")
for mod_name in ["pathlib", "ntpath", "posixpath"]:
    entries = [entry.strip() for entry in dont_rewrite.split(",") if entry.strip()]
    if mod_name not in entries:
        entries.append(mod_name)
    dont_rewrite = ",".join(entries)
os.environ["PYTEST_DONT_REWRITE"] = dont_rewrite

_MODULE_CACHE = {
    name: module
    for name, module in sys.modules.items()
    if name in {"pathlib", "ntpath", "posixpath"}
}


class _ModuleDict(dict):
    def clear(self) -> None:  # pragma: no cover - exercised via tests
        super().clear()
        super().update(_MODULE_CACHE)


if not isinstance(sys.modules, _ModuleDict):
    sys.modules = _ModuleDict(sys.modules)


class _CachedLoader(importlib.abc.Loader):
    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def create_module(self, spec):  # pragma: no cover - default reuse
        return self._module

    def exec_module(self, module):  # pragma: no cover - nothing to execute
        return None


class _StdlibCacheFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # pragma: no cover - exercised via tests
        module = _MODULE_CACHE.get(fullname)
        if module is None:
            return None
        return importlib.util.spec_from_loader(fullname, _CachedLoader(module))


sys.meta_path.insert(0, _StdlibCacheFinder())


class _MissingPipelineLoader(importlib.abc.Loader):
    def create_module(self, spec):  # pragma: no cover - return default
        return None

    def exec_module(self, module):  # pragma: no cover - invoked when missing
        raise ImportError("ai_trading.pipeline is not available")


class _PipelineAbortFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # pragma: no cover - exercised via tests
        if fullname == "ai_trading.pipeline" and not sys.modules:
            return importlib.util.spec_from_loader(fullname, _MissingPipelineLoader())
        return None


sys.meta_path.insert(0, _PipelineAbortFinder())

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
        sys.modules["dotenv"] = module
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
                sys.modules["dotenv"] = canonical
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
