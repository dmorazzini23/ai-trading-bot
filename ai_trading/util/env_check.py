"""Environment import guards to ensure :mod:`python-dotenv` resolves correctly."""

from __future__ import annotations

import logging
import os
import sys
from importlib import import_module, util as importlib_util
from pathlib import Path


logger = logging.getLogger(__name__)


class DotenvImportError(ImportError):
    """Raised when :mod:`python-dotenv` is missing or shadowed by a local module."""


_FALSEY = {"0", "false", "no", "off", "disable", "disabled"}


def _guard_enabled(force: bool | None = None) -> bool:
    if force is not None:
        return bool(force)
    raw = os.getenv("ENV_IMPORT_GUARD", "true")
    return raw.strip().lower() not in _FALSEY if isinstance(raw, str) else bool(raw)


def guard_python_dotenv(*, force: bool | None = None) -> None:
    """Validate that :mod:`python-dotenv` resolves from site/dist-packages."""

    if not _guard_enabled(force):
        logger.debug("ENV_IMPORT_GUARD_DISABLED")
        return

    existing = sys.modules.get("dotenv")
    if existing is not None and getattr(existing, "__spec__", None) is None:
        sys.modules.pop("dotenv", None)

    try:
        spec = importlib_util.find_spec("dotenv")
    except ValueError as exc:
        raise DotenvImportError("python-dotenv spec resolution failed") from exc
    if spec is None or not getattr(spec, "origin", None):
        raise DotenvImportError("python-dotenv not found; install python-dotenv>=1.1.1")

    origin = Path(spec.origin).resolve()
    parents = list(origin.parents)
    in_site_packages = any(
        part.name.endswith("site-packages") or part.name.endswith("dist-packages")
        for part in parents
    )

    if not in_site_packages:
        raise DotenvImportError(f"python-dotenv appears shadowed; origin={origin}")

    repo_markers = {"ai_trading", "tests", "workspace", "src"}
    if any(part.name in repo_markers for part in parents):
        raise DotenvImportError(f"python-dotenv resolved from project tree: origin={origin}")

    sys.modules.pop("dotenv", None)
    try:
        module = import_module("dotenv")
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise DotenvImportError("python-dotenv import failed") from exc

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise DotenvImportError("python-dotenv missing __file__ metadata")

    module_path = Path(module_file).resolve()
    if module_path != origin:
        # Some environments may load package __init__ while spec points to compiled file;
        # accept as long as both live under site-packages.
        if module_path not in origin.parents and origin not in module_path.parents:
            raise DotenvImportError(
                f"python-dotenv mismatch: spec={origin} import={module_path}"
            )

    if not hasattr(module, "dotenv_values"):
        raise DotenvImportError(
            "python-dotenv shadowed by another module named 'dotenv'."
        )

    logger.debug("ENV_IMPORT_GUARD_OK", extra={"module": str(module_path)})


def main() -> None:
    """CLI entry point for quick validation in CI environments."""

    guard_python_dotenv()


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    main()
