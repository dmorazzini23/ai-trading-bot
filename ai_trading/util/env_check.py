import importlib.util
from pathlib import Path


class DotenvImportError(ImportError):
    pass


PYTHON_DOTENV_RESOLVED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def ensure_python_dotenv_is_real_package() -> None:
    """Raise if ``dotenv`` resolves to a shadowed in-repo package."""

    spec = importlib.util.find_spec("dotenv")
    if not spec or not getattr(spec, "origin", None):
        return

    origin = Path(spec.origin).resolve()
    if _is_under(origin, _repo_root()):
        raise DotenvImportError(f"Refusing to import shadowed dotenv at {origin}")

    globals()["PYTHON_DOTENV_RESOLVED"] = True


def assert_dotenv_not_shadowed() -> None:
    ensure_python_dotenv_is_real_package()
