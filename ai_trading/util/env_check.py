import importlib
import importlib.machinery
import importlib.util
from pathlib import Path


class DotenvImportError(ImportError):
    pass


PYTHON_DOTENV_RESOLVED: bool | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_site_packages(path: Path) -> bool:
    """Return True if ``path`` resides in a site-packages/dist-packages directory."""

    site_dirs = {"site-packages", "dist-packages"}
    return any(part.lower() in site_dirs for part in path.resolve().parts)


def ensure_python_dotenv_is_real_package() -> None:
    """Raise if ``dotenv`` resolves to a shadowed in-repo package."""

    guard_flag = globals().get("PYTHON_DOTENV_RESOLVED", None)
    if guard_flag is False:
        return

    repo_root = _repo_root()
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
    if spec is not None and getattr(spec, "origin", None):
        try:
            spec_origin = Path(spec.origin).resolve()
        except Exception:
            spec_origin = None
    if spec_origin is not None:
        origin = spec_origin

    if origin is None:
        globals()["PYTHON_DOTENV_RESOLVED"] = True
        return

    if "/tests/stubs/dotenv/" in str(origin):
        globals()["PYTHON_DOTENV_RESOLVED"] = True
        return

    shadow_root = (repo_root / "dotenv").resolve()
    if origin.is_relative_to(shadow_root):
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        raise DotenvImportError(f"python-dotenv is shadowed at {origin}")

    if origin.name == "dotenv.py" and origin.is_relative_to(repo_root):
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        raise DotenvImportError(f"python-dotenv is shadowed at {origin}")

    if _is_site_packages(origin):
        globals()["PYTHON_DOTENV_RESOLVED"] = True
        return

    globals()["PYTHON_DOTENV_RESOLVED"] = True


def guard_python_dotenv_import() -> None:
    """Import ``dotenv`` and ensure it does not resolve to a shadowed module."""

    ensure_python_dotenv_is_real_package()


def guard_dotenv_shadowing() -> None:
    """Alias for :func:`guard_python_dotenv_import` for compatibility."""

    ensure_python_dotenv_is_real_package()


def assert_dotenv_not_shadowed() -> None:
    ensure_python_dotenv_is_real_package()
