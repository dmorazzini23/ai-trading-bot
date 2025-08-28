"""Minimal `requests.sessions` stub for tests.

This module provides a tiny subset of the real ``requests.sessions``
interface so that code importing ``requests.sessions.Session`` continues to
work even when the actual ``requests`` library is not installed. Network
operations are not performed.
"""
from __future__ import annotations

from typing import Any

from . import ConnectionError


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


__all__ = ["Session", "Response"]

