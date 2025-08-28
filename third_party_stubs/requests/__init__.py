"""Minimal `requests` stub used in tests when the real library is absent.

Only a tiny subset of the API is implemented to satisfy the project's
imports. Network operations are not performed.
"""
from __future__ import annotations

from typing import Any


class RequestException(Exception):
    """Base exception for request errors."""


class HTTPError(RequestException):
    """Raised for HTTP errors."""

    def __init__(self, *args: Any, response: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args)
        self.response = response


class ConnectionError(RequestException):
    """Raised when a connection error occurs."""


# exceptions submodule mimicking real requests structure
exceptions = type(
    "exceptions",
    (),
    {
        "RequestException": RequestException,
        "HTTPError": HTTPError,
        "ConnectionError": ConnectionError,
    },
)


from .sessions import Session, Response


__all__ = [
    "RequestException",
    "HTTPError",
    "ConnectionError",
    "Session",
    "Response",
    "exceptions",
]
