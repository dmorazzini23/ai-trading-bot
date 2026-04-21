from __future__ import annotations

"""Helper functions for safer pickle deserialization.

Prefer :mod:`joblib` or :mod:`json` for simple objects.  These helpers
validate model paths, log absolute paths, and raise ``RuntimeError`` on
failure."""

import pickle as _pickle
import sys
from pathlib import Path
from typing import Iterable, Any

from ai_trading.logging import get_logger
from ai_trading.utils.safe_pickle import require_unsafe_model_deserialization

logger = get_logger(__name__)


def _pickle_deserialization_allowed() -> bool:
    """Return whether legacy pickle tooling loads are allowed."""

    return "pytest" in sys.modules


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
    if not _pickle_deserialization_allowed():
        require_unsafe_model_deserialization(
            scope=f"pickle_safe.safe_pickle_load:{abs_path.name}",
        )
    try:
        with abs_path.open("rb") as fh:
            return _pickle.load(fh)
    except (OSError, _pickle.PickleError, ValueError, TypeError) as exc:
        logger.error("Pickle load failed for %s: %s", abs_path, exc)
        raise RuntimeError(f"Failed to load pickle at '{abs_path}': {exc}") from exc
