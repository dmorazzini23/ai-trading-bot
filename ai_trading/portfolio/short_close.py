from __future__ import annotations

"""Utilities for closing out short positions."""

from collections.abc import Callable, Iterable
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def short_close(api: Any, submit: Callable[[str, int, str], Any]) -> int:
    """Close any open short positions using provided broker API.

    Parameters
    ----------
    api:
        Broker API instance with ``list_positions`` method.
    submit:
        Callable used to submit buy orders. It must accept ``symbol``,
        ``quantity`` and ``side`` arguments.

    Returns
    -------
    int
        Number of close orders submitted.
    """
    positions: Iterable[Any] = []
    if hasattr(api, "list_positions"):
        try:
            positions = api.list_positions() or []
        except Exception:  # pragma: no cover - defensive
            logger.warning("LIST_POSITIONS_FAIL", exc_info=True)
            positions = []
    submitted = 0
    for p in positions:
        try:
            qty = int(float(getattr(p, "qty", 0)))
        except (TypeError, ValueError):
            continue
        if qty < 0:
            symbol = getattr(p, "symbol", None)
            if symbol:
                submit(symbol, abs(qty), "buy")
                submitted += 1
                logger.info("SHORT_CLOSE_SUBMIT | symbol=%s qty=%d", symbol, abs(qty))
    return submitted
