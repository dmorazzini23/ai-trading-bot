from __future__ import annotations

"""Minimal urllib3.exceptions stub for tests."""

class HTTPWarning(Warning):
    """Base warning type for HTTP issues."""


class SystemTimeWarning(HTTPWarning):
    """Warning for TLS clock skew issues."""


class HTTPError(Exception):
    """Placeholder matching urllib3's HTTPError."""


class DependencyWarning(Warning):
    """Dependency mismatch warning stub."""


__all__ = ["HTTPWarning", "SystemTimeWarning", "HTTPError", "DependencyWarning"]

_DYNAMIC_CACHE: dict[str, type] = {}


def __getattr__(name: str):
    if name in _DYNAMIC_CACHE:
        return _DYNAMIC_CACHE[name]
    base = Warning if name.endswith("Warning") else Exception
    cls = type(name, (base,), {})
    _DYNAMIC_CACHE[name] = cls
    return cls
