from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ai_trading.data.bars import BarsFetchFailed

from ai_trading.net.http import HTTPSession
from ai_trading.integrations.rate_limit import get_rate_limiter


_rate_limiter = get_rate_limiter()


def fetch(
    url: str,
    *,
    session: Any | None = None,
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

    response = session.get(url, **kwargs)

    bars_fetch_failed_cls: type[BaseException] | None = None

    # Import lazily to avoid circular dependencies during module import.
    try:  # pragma: no cover - fast path when bars available
        from ai_trading.data.bars import BarsFetchFailed as _bars_fetch_failed_cls
        bars_fetch_failed_cls = cast(type[BaseException], _bars_fetch_failed_cls)
    except Exception:  # pragma: no cover - defensive guard
        bars_fetch_failed_cls = None

    if bars_fetch_failed_cls is not None and isinstance(response, bars_fetch_failed_cls):
        return response

    return response


__all__ = ["fetch"]
