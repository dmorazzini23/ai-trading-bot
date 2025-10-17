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
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        return

    repo_root = _repo_root()
    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        spec = importlib.machinery.PathFinder.find_spec("dotenv")

    if spec is None:
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        return

    origin_str = getattr(spec, "origin", None)
    if origin_str is None:
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        return

    origin = Path(origin_str).resolve()

    if "/tests/stubs/dotenv/" in str(origin):
        globals()["PYTHON_DOTENV_RESOLVED"] = True
        return

    shadow_root = (repo_root / "dotenv").resolve()
    if origin.is_relative_to(shadow_root):
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        raise DotenvImportError(f"python-dotenv is shadowed at {origin}")

    if _is_site_packages(origin):
        globals()["PYTHON_DOTENV_RESOLVED"] = True
        return

    globals()["PYTHON_DOTENV_RESOLVED"] = True


def assert_dotenv_not_shadowed() -> None:
    ensure_python_dotenv_is_real_package()
