import importlib
import importlib.machinery
import pathlib
import sys

from ai_trading.env import PYTHON_DOTENV_RESOLVED


class DotenvImportError(ImportError): ...


def assert_dotenv_not_shadowed():
    """
    Raise DotenvImportError if the import resolver would load a 'dotenv'
    module from inside this repository (shadowing site-packages). Tests monkeypatch
    importlib to simulate this; we must consult find_spec, not sys.modules.
    """
    import importlib.util
    import os
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.find_spec("dotenv")
    if spec is None or not getattr(spec, "origin", None):
        return
    origin = Path(spec.origin).resolve()

    # Accept typical virtualenv/site-packages paths
    allow = ("site-packages", "dist-packages", ".venv", "venv", "env")
    if any(part in origin.parts for part in allow):
        return

    # If origin sits under the repository root, fail fast
    try:
        origin.relative_to(repo_root)
        raise DotenvImportError(f"dotenv import resolves inside repo: {origin}")
    except ValueError:
        # Different tree; OK
        return
