"""Lightweight urllib3 stub for tests."""

from __future__ import annotations

import warnings

def _load_retry() -> object | None:
    try:
        from .util.retry import Retry as loaded_retry  # noqa: F401
        return loaded_retry
    except Exception:  # pragma: no cover - missing util stub
        return None


Retry = _load_retry()

__version__ = "2.0.0"

from . import exceptions as exceptions  # re-export module for import parity
from .exceptions import HTTPError, HTTPWarning, SystemTimeWarning


def disable_warnings(category: type[Warning] | None = HTTPWarning) -> None:
    """Mimic urllib3.disable_warnings by silencing the provided warning."""

    if category is None:
        category = HTTPWarning
    try:
        warnings.filterwarnings("ignore", category=category)
    except Exception:
        # In line with urllib3's behavior, failures should not raise.
        pass


__all__ = [
    "Retry",
    "exceptions",
    "disable_warnings",
    "HTTPWarning",
    "SystemTimeWarning",
    "HTTPError",
]
