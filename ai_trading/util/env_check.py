"""Environment import guards to ensure expected third-party modules are loaded."""

from __future__ import annotations

import logging
import os
import sys
from importlib import import_module


logger = logging.getLogger(__name__)


class DotenvImportError(ImportError):
    """Raised when :mod:`python-dotenv` is shadowed by an unexpected module."""


_FALSEY = {"0", "false", "no", "off", "disable", "disabled"}


def _guard_enabled(force: bool | None = None) -> bool:
    if force is not None:
        return bool(force)
    raw = os.getenv("ENV_IMPORT_GUARD", "true")
    return raw.strip().lower() not in _FALSEY if isinstance(raw, str) else bool(raw)


def guard_python_dotenv(*, force: bool | None = None) -> None:
    """Validate that :mod:`python-dotenv` is importable and not shadowed."""

    if not _guard_enabled(force):
        logger.debug("ENV_IMPORT_GUARD_DISABLED")
        return

    sys.modules.pop("dotenv", None)
    try:
        module = import_module("dotenv")
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise ImportError("python-dotenv not installed; install python-dotenv>=1.1.1") from exc

    module_file = getattr(module, "__file__", "unknown")
    module_spec = getattr(module, "__spec__", None)
    has_attr = hasattr(module, "dotenv_values")
    if not has_attr:
        logger.debug(
            "ENV_IMPORT_GUARD_FAILURE",
            extra={
                "module": module_file,
                "module_repr": repr(module),
                "module_spec": repr(module_spec),
            },
        )
        raise DotenvImportError(
            "python-dotenv not available (shadowed by 'dotenv'). "
            "Uninstall 'dotenv' package and remove any local module named 'dotenv'. "
            f"loaded={module_file!r} module={module!r} spec={module_spec!r}"
        )

    logger.debug("ENV_IMPORT_GUARD_OK", extra={"module": module_file})


def main() -> None:
    """CLI entry point for quick validation in CI environments."""

    guard_python_dotenv()


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    main()
