from __future__ import annotations

"""Helper functions for safer pickle deserialization.

Prefer :mod:`joblib` or :mod:`json` for simple objects.  These helpers
validate model paths, log absolute paths, and raise ``RuntimeError`` on
failure."""

import pickle
from pathlib import Path
from typing import Iterable, Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def safe_pickle_load(path: Path, allowed_dirs: Iterable[Path]) -> Any:
    """Load a pickle file after validating ``path``.

    Parameters
    ----------
    path:
        Location of the pickle file.
    allowed_dirs:
        Iterable of directories that ``path`` must reside in.
    """
    abs_path = path.resolve()
    allowed = [d.resolve() for d in allowed_dirs]
    if not any(abs_path.is_relative_to(d) for d in allowed):
        raise RuntimeError(
            f"Attempted to load pickle outside allowed directories: {abs_path}"
        )
    try:
        with abs_path.open("rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.PickleError, ValueError, TypeError) as exc:
        logger.error("Pickle load failed for %s: %s", abs_path, exc)
        raise RuntimeError(f"Failed to load pickle at '{abs_path}': {exc}") from exc
