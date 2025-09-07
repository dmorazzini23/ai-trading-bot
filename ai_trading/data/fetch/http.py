from __future__ import annotations

from typing import Any

from . import _HTTP_SESSION, _SIP_UNAUTHORIZED
from ai_trading.net.http import HTTPSession


def get(
    url: str,
    *,
    feed: str | None = None,
    session: HTTPSession | None = None,
    **kwargs: Any,
) -> Any:
    """Issue a ``GET`` request while enforcing SIP authorization.

    Parameters
    ----------
    url:
        Target URL to request.
    feed:
        Data feed being accessed, e.g. ``"sip"`` or ``"iex"``.
    session:
        Optional HTTP session used to issue the request. Falls back to the
        package-level session when omitted.

    Returns
    -------
    Any
        Response object returned by ``session.get``.

    Raises
    ------
    ValueError
        If SIP access is unauthorized or the session is missing/invalid.
    """
    if feed == "sip" and _SIP_UNAUTHORIZED:
        raise ValueError("sip_unauthorized")

    session = session or _HTTP_SESSION
    if session is None or not hasattr(session, "get"):
        raise ValueError("session_required")

    return session.get(url, **kwargs)


__all__ = ["get"]
