"""Lightweight proxy for :mod:`ai_trading.alpaca_api` used by tests."""

from typing import Any

from ai_trading.alpaca_api import *  # noqa: F401,F403

_SUBMIT_ORDER_IMPL = globals().get("submit_order")


def submit_order(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
    """Backwards-compatible ``submit_order`` shim accepting legacy positional qty."""

    if _SUBMIT_ORDER_IMPL is None:
        raise AttributeError("submit_order not available")
    if len(args) >= 3 and "qty" not in kwargs:
        symbol, candidate_qty, candidate_side = args[:3]
        if isinstance(candidate_side, str) and not isinstance(candidate_qty, str):
            remaining = args[3:]
            return _SUBMIT_ORDER_IMPL(
                symbol,
                candidate_side,
                *remaining,
                qty=candidate_qty,
                **kwargs,
            )
    return _SUBMIT_ORDER_IMPL(*args, **kwargs)


DRY_RUN = True


__all__ = [name for name in globals() if not name.startswith("_")]
