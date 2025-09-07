from __future__ import annotations

from typing import Any

from ai_trading.net.http import HTTPSession


def fetch(url: str, *, session: HTTPSession | None = None, **kwargs: Any) -> Any:
    """Fetch ``url`` using the provided HTTP session.

    Parameters
    ----------
    url:
        Target URL to request.
    session:
        HTTP session used to issue the request. Must provide a ``get`` method.

    Returns
    -------
    Any
        The response returned by ``session.get``.

    Raises
    ------
    ValueError
        If ``session`` is ``None`` or lacks a ``get`` method.
    """
    if session is None or not hasattr(session, "get"):
        raise ValueError("session_required")

    return session.get(url, **kwargs)


__all__ = ["fetch"]
