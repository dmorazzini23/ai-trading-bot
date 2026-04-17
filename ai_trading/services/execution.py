from __future__ import annotations

from json import JSONDecodeError
from typing import Any


def execute_signal_orders(
    ctx: Any,
    signals: Any,
    *,
    logger: Any,
) -> list[tuple[str, str]]:
    """Convert directional signals into simple orders without broker coupling."""

    orders: list[tuple[str, str]] = []
    items = getattr(signals, "items", None)
    if not callable(items):
        return orders
    for symbol, sig in items():
        if sig == 0:
            continue
        side = "buy" if sig > 0 else "sell"
        api = getattr(ctx, "api", None)
        if api is not None and hasattr(api, "submit_order"):
            try:
                api.submit_order(symbol, 1, side)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                JSONDecodeError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
            ) as exc:
                logger.error(
                    "Failed to submit test order for %s %s: %s",
                    symbol,
                    side,
                    exc,
                )
        orders.append((str(symbol), side))
    return orders


__all__ = ["execute_signal_orders"]
