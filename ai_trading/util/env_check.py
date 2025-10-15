import importlib
import importlib.machinery
import pathlib
import importlib.util
from pathlib import Path


class DotenvImportError(ImportError):
    pass


PYTHON_DOTENV_RESOLVED = False


def ensure_python_dotenv_is_real_package() -> None:
    """Raise if ``dotenv`` resolves to a shadowed in-repo package."""

    spec = importlib.util.find_spec("dotenv")
    if not spec or not getattr(spec, "origin", None):
        return

    origin = Path(spec.origin).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root in origin.parents:
        raise DotenvImportError(f"Refusing to import shadowed dotenv at {origin}")


def assert_dotenv_not_shadowed() -> None:
    ensure_python_dotenv_is_real_package()
