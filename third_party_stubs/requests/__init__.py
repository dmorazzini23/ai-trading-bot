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


class Response:
    """Placeholder response object used in tests."""

    status_code: int = 200
    text: str = ""

    def json(self) -> Any:  # pragma: no cover - defensive
        raise ValueError("No JSON in stub response")


class Session:
    """Very small subset of :class:`requests.Session`."""

    def request(self, method: str, url: str, **kwargs: Any) -> Response:  # pragma: no cover - deterministic
        """Perform an HTTP request.

        The stub does not perform network I/O and always raises
        :class:`ConnectionError` to mirror a failed request.
        """
        raise ConnectionError("requests stub cannot perform HTTP operations")

    def get(self, url: str, **kwargs: Any) -> Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response:
        return self.request("POST", url, **kwargs)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - no cleanup
        return False

    # Methods expected by tests that patch Session
    def close(self) -> None:  # pragma: no cover - no state to close
        return None

    def mount(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        return None


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


# Re-export common classes at the package root
RequestException = RequestException
HTTPError = HTTPError
ConnectionError = ConnectionError
Session = Session
Response = Response

__all__ = [
    "RequestException",
    "HTTPError",
    "ConnectionError",
    "Session",
    "Response",
    "exceptions",
]
