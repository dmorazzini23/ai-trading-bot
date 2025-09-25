import importlib, sys, pathlib


class DotenvImportError(ImportError): ...


def assert_dotenv_not_shadowed():
    existing = sys.modules.get("dotenv")
    if existing is not None and getattr(existing, "__spec__", None) is None:
        sys.modules.pop("dotenv", None)
    try:
        spec = importlib.util.find_spec("dotenv")
    except ValueError as exc:
        raise DotenvImportError("python-dotenv not importable") from exc
    if spec is None or not spec.origin:
        raise DotenvImportError("python-dotenv not importable")
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if pathlib.Path(spec.origin).resolve().is_relative_to(repo_root):
        raise DotenvImportError(f"python-dotenv is shadowed by {spec.origin}")
