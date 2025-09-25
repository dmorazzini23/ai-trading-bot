"""Lightweight urllib3 stub for tests."""

from __future__ import annotations

import warnings

try:
    from .util.retry import Retry  # noqa: F401  # re-export for import parity
except Exception:  # pragma: no cover - missing util stub
    Retry = None  # type: ignore[assignment]

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
