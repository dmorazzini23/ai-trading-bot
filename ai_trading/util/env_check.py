import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


class DotenvImportError(ImportError):
    pass


PYTHON_DOTENV_RESOLVED: bool | None = None
_ASSERT_GUARD_DISABLED: bool = False
_ASSERT_GUARD_DISABLED_FOR: str | None = None


def _managed_env_value(name: str) -> str | None:
    for module_name in ("ai_trading.config", "ai_trading.config.management"):
        module = sys.modules.get(module_name)
        getter = getattr(module, "get_env", None) if module is not None else None
        if not callable(getter):
            continue
        try:
            value = getter(name, None, cast=str, resolve_aliases=False)
        except Exception:
            continue
        if value in (None, ""):
            continue
        return str(value)
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_site_packages(path: Path) -> bool:
    """Return True if ``path`` resides in a site-packages/dist-packages directory."""

    site_dirs = {"site-packages", "dist-packages"}
    return any(part.lower() in site_dirs for part in path.resolve().parts)


def _resolve_spec_origin() -> Path | None:
    """Return the resolved ``Path`` for the ``dotenv`` import spec if available."""

    finders = (
        importlib.util.find_spec,
        importlib.machinery.PathFinder.find_spec,
    )
    for finder in finders:
        try:
            spec = finder("dotenv")
        except Exception:
            continue
        if spec is None:
            continue
        origin = getattr(spec, "origin", None)
        if not origin:
            continue
        try:
            return Path(origin).resolve()
        except Exception:
            continue
    return None


def _is_shadowed_path(origin: Path, repo_root: Path) -> bool:
    """Return ``True`` when *origin* points at a repo-local dotenv shadow."""

    resolved = origin.resolve()
    resolved_str = str(resolved)
    repo_root_str = str(repo_root.resolve())

    # Allow pytest stubs that intentionally shadow the package.
    if "/tests/stubs/dotenv/" in resolved_str:
        return False
    if _is_site_packages(resolved):
        return False

    shadow_dir = (repo_root / "dotenv").resolve()
    shadow_file = (repo_root / "dotenv.py").resolve()
    if resolved == shadow_file:
        return True
    if resolved == shadow_dir:
        return True
    if shadow_dir in resolved.parents:
        return True

    if resolved_str.startswith(repo_root_str):
        if resolved_str.endswith("dotenv.py"):
            return True
        if f"{repo_root_str}/dotenv/" in resolved_str:
            return True
    return False


def ensure_python_dotenv_is_real_package() -> None:
    """Raise if ``dotenv`` resolves to a shadowed in-repo package."""

    guard_flag = globals().get("PYTHON_DOTENV_RESOLVED", None)
    if guard_flag is False:
        return

    repo_root = _repo_root()
    resolved_spec_origin = _resolve_spec_origin()
    if resolved_spec_origin and _is_shadowed_path(resolved_spec_origin, repo_root):
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        raise DotenvImportError(f"python-dotenv is shadowed at {resolved_spec_origin}")

    try:
        module = importlib.import_module("dotenv")
    except ImportError as exc:
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        raise DotenvImportError("python-dotenv could not be imported") from exc

    origin_attr = getattr(module, "__file__", None)
    if origin_attr is None:
        origin = None
    else:
        origin = Path(origin_attr).resolve()

    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        spec = importlib.machinery.PathFinder.find_spec("dotenv")
    spec_origin: Path | None = None
    raw_spec_origin = getattr(spec, "origin", None) if spec is not None else None
    if isinstance(raw_spec_origin, str) and raw_spec_origin:
        try:
            spec_origin = Path(raw_spec_origin).resolve()
        except Exception:
            spec_origin = None
    if spec_origin is not None:
        origin = spec_origin

    if origin is None:
        globals()["PYTHON_DOTENV_RESOLVED"] = True
        return
    if _is_shadowed_path(origin, repo_root):
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        raise DotenvImportError(f"python-dotenv is shadowed at {origin}")

    globals()["PYTHON_DOTENV_RESOLVED"] = True


def guard_python_dotenv_import() -> None:
    """Import ``dotenv`` and ensure it does not resolve to a shadowed module."""

    ensure_python_dotenv_is_real_package()


def guard_dotenv_shadowing() -> None:
    """Alias for :func:`guard_python_dotenv_import` for compatibility."""

    ensure_python_dotenv_is_real_package()


def assert_dotenv_not_shadowed() -> None:
    global _ASSERT_GUARD_DISABLED, _ASSERT_GUARD_DISABLED_FOR

    current_test = _managed_env_value("PYTEST_CURRENT_TEST")
    if _ASSERT_GUARD_DISABLED and _ASSERT_GUARD_DISABLED_FOR is not None:
        if current_test != _ASSERT_GUARD_DISABLED_FOR:
            _ASSERT_GUARD_DISABLED = False
            _ASSERT_GUARD_DISABLED_FOR = None

    if _ASSERT_GUARD_DISABLED:
        return
    ensure_python_dotenv_is_real_package()


_CANONICAL_ASSERT_GUARD = assert_dotenv_not_shadowed


def disable_dotenv_guard() -> None:
    """Disable the import shadow guard until explicitly re-enabled."""

    global _ASSERT_GUARD_DISABLED, _ASSERT_GUARD_DISABLED_FOR
    _ASSERT_GUARD_DISABLED = True
    _ASSERT_GUARD_DISABLED_FOR = _managed_env_value("PYTEST_CURRENT_TEST")


def enable_dotenv_guard() -> None:
    """Re-enable the import shadow guard."""

    global _ASSERT_GUARD_DISABLED, _ASSERT_GUARD_DISABLED_FOR
    _ASSERT_GUARD_DISABLED = False
    _ASSERT_GUARD_DISABLED_FOR = None


class _EnvCheckModule(types.ModuleType):
    """Module wrapper that interprets guard overrides as on/off toggles."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "assert_dotenv_not_shadowed":
            canonical = object.__getattribute__(self, "_CANONICAL_ASSERT_GUARD")
            if value is canonical:
                object.__setattr__(self, "_ASSERT_GUARD_DISABLED", False)
                object.__setattr__(self, "_ASSERT_GUARD_DISABLED_FOR", None)
            else:
                object.__setattr__(self, "_ASSERT_GUARD_DISABLED", True)
                object.__setattr__(self, "_ASSERT_GUARD_DISABLED_FOR", _managed_env_value("PYTEST_CURRENT_TEST"))
            super().__setattr__(name, canonical)
            super().__setattr__("_ASSERT_GUARD_DISABLED", object.__getattribute__(self, "_ASSERT_GUARD_DISABLED"))
            super().__setattr__("_ASSERT_GUARD_DISABLED_FOR", object.__getattribute__(self, "_ASSERT_GUARD_DISABLED_FOR"))
            return
        super().__setattr__(name, value)


_module = sys.modules.get(__name__)
if _module is not None and not isinstance(_module, _EnvCheckModule):
    proxy = _EnvCheckModule(__name__)
    proxy.__dict__.update(_module.__dict__)
    object.__setattr__(proxy, "_CANONICAL_ASSERT_GUARD", _module.__dict__["assert_dotenv_not_shadowed"])
    object.__setattr__(proxy, "_ASSERT_GUARD_DISABLED", _module.__dict__.get("_ASSERT_GUARD_DISABLED", False))
    object.__setattr__(proxy, "_ASSERT_GUARD_DISABLED_FOR", _module.__dict__.get("_ASSERT_GUARD_DISABLED_FOR", None))
    sys.modules[__name__] = proxy


__all__ = [
    "PYTHON_DOTENV_RESOLVED",
    "assert_dotenv_not_shadowed",
    "disable_dotenv_guard",
    "enable_dotenv_guard",
    "guard_dotenv_shadowing",
    "guard_python_dotenv_import",
]
