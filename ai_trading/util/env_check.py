import importlib
import importlib.machinery
import pathlib
import sys

from ai_trading.env import PYTHON_DOTENV_RESOLVED


class DotenvImportError(ImportError): ...


def assert_dotenv_not_shadowed():
    if not PYTHON_DOTENV_RESOLVED:
        return

    existing = sys.modules.get("dotenv")
    if existing is not None and getattr(existing, "__spec__", None) is None:
        sys.modules.pop("dotenv", None)
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    stub_path = repo_root / "tests" / "stubs"
    search_paths = [p for p in sys.path if pathlib.Path(p).resolve() != stub_path]

    try:
        spec = importlib.machinery.PathFinder.find_spec("dotenv", search_paths)
    except ValueError as exc:
        raise DotenvImportError("python-dotenv not importable") from exc
    if spec is None:
        try:
            spec = importlib.util.find_spec("dotenv")
        except ValueError as exc:
            raise DotenvImportError("python-dotenv not importable") from exc
    if spec is None or not spec.origin:
        raise DotenvImportError("python-dotenv not importable")
    origin_path = pathlib.Path(spec.origin).resolve()
    try:
        relative_origin = origin_path.relative_to(repo_root)
    except ValueError:
        # Installed outside of the repository – definitely safe.
        return

    # Virtual environments that live inside the repository (e.g. "venv" or
    # ".venv") install third-party packages under ``site-packages``. When
    # python-dotenv is loaded from there we do not want to treat it as a
    # shadowing module.
    parts = set(relative_origin.parts)
    if "site-packages" in parts:
        return

    # Allow common in-repo virtual environment directory prefixes as well.
    if relative_origin.parts and relative_origin.parts[0] in {"venv", ".venv", "env", ".env"}:
        return

    raise DotenvImportError(f"python-dotenv is shadowed by {spec.origin}")
