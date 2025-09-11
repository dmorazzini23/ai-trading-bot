from __future__ import annotations

from typing import Any

from ai_trading.net.http import HTTPSession
from ai_trading.integrations.rate_limit import get_rate_limiter


_rate_limiter = get_rate_limiter()


def fetch(
    url: str,
    *,
    session: HTTPSession | None = None,
    route: str = "bars",
    **kwargs: Any,
) -> Any:
    """Fetch ``url`` using the provided HTTP session.

    Parameters
    ----------
    url:
        Target URL to request.
    session:
        HTTP session used to issue the request. Must provide a ``get`` method.
    route:
        Identifier for rate limiting. Defaults to ``"bars"``.

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

    if not _rate_limiter.acquire_sync(route):
        raise RuntimeError("rate_limit_exceeded")

    return session.get(url, **kwargs)


__all__ = ["fetch"]
