import importlib.util
from pathlib import Path


class DotenvImportError(ImportError):
    pass


PYTHON_DOTENV_RESOLVED: bool | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_site_packages(path: Path) -> bool:
    """Return True if ``path`` resides in a site-packages/dist-packages directory."""

    site_dirs = {"site-packages", "dist-packages"}
    return any(part.lower() in site_dirs for part in path.resolve().parts)


def ensure_python_dotenv_is_real_package() -> None:
    """Raise if ``dotenv`` resolves to a shadowed in-repo package."""

    guard_flag = globals().get("PYTHON_DOTENV_RESOLVED", None)

    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        if guard_flag is False:
            globals()["PYTHON_DOTENV_RESOLVED"] = False
            return
        raise DotenvImportError("python-dotenv is not installed")

    if guard_flag is False:
        globals()["PYTHON_DOTENV_RESOLVED"] = False
        return

    candidates: list[Path] = []

    origin = getattr(spec, "origin", None)
    if origin:
        try:
            candidates.append(Path(origin).resolve())
        except Exception:
            pass

    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        for entry in locations:
            try:
                candidates.append(Path(entry).resolve())
            except Exception:
                continue

    if not candidates:
        raise DotenvImportError("Unable to locate python-dotenv package")

    repo_root = _repo_root()
    for candidate in candidates:
        if _is_under(candidate, repo_root) and not _is_site_packages(candidate):
            raise DotenvImportError(
                f"python-dotenv is shadowed at {candidate}"
            )

    globals()["PYTHON_DOTENV_RESOLVED"] = True


def assert_dotenv_not_shadowed() -> None:
    ensure_python_dotenv_is_real_package()
