"""Lightweight urllib3 stub for tests."""

from __future__ import annotations

import types
import warnings

try:
    from .util.retry import Retry  # noqa: F401  # re-export for import parity
except Exception:  # pragma: no cover - missing util stub
    Retry = None  # type: ignore[assignment]


class HTTPWarning(Warning):
    """Base warning type for HTTP issues."""


class SystemTimeWarning(HTTPWarning):
    """Warning for TLS clock skew issues."""


exceptions = types.SimpleNamespace(
    HTTPWarning=HTTPWarning,
    SystemTimeWarning=SystemTimeWarning,
)


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
]
