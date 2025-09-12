from __future__ import annotations

"""Heartbeat helpers with provider fallback."""

from typing import Any, Callable

from ai_trading.logging import get_logger

from .fallback_order import mark_yahoo

logger = get_logger(__name__)


def heartbeat(primary: Callable[[], Any], fallback: Callable[[], Any]) -> Any:
    """Execute ``primary`` heartbeat, falling back on failure.

    Parameters
    ----------
    primary:
        Callable representing the primary heartbeat action.
    fallback:
        Callable invoked when ``primary`` raises an exception.

    Returns
    -------
    Any
        The result of ``primary`` when it succeeds, otherwise the
        result of ``fallback``.
    """
    try:
        return primary()
    except Exception:  # pragma: no cover - best effort
        logger.warning("HEARTBEAT_FALLBACK_TRIGGERED")
        mark_yahoo()
        return fallback()


__all__ = ["heartbeat"]
