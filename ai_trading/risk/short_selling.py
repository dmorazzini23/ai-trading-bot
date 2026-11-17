"""Short selling validation utilities."""
from __future__ import annotations

from typing import Any


SHORT_SIDE_ALIASES = frozenset(
    {
        "sell_short",
        "sellshort",
        "short",
        "sell-short",
        "sell short",
        "short_sell",
        "short-sell",
        "short sell",
        "enter_short",
        "entry_short",
        "open_short",
        "sell_to_open",
        "sell-to-open",
        "sell to open",
    }
)


def normalize_short_side(side: Any) -> str:
    token = str(side or "").strip().lower()
    return "sell_short" if token in SHORT_SIDE_ALIASES else token


def is_short_side(side: Any) -> bool:
    return normalize_short_side(side) == "sell_short"


def _read_attr(record: Any, *names: str) -> Any:
    for name in names:
        if isinstance(record, dict) and name in record:
            return record.get(name)
        value = getattr(record, name, None)
        if value is not None:
            return value
    return None


def _bool_from_record(record: Any, *names: str) -> bool | None:
    value = _read_attr(record, *names)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on", "enabled", "available"}:
        return True
    if token in {"0", "false", "no", "n", "off", "disabled", "unavailable"}:
        return False
    return None


def validate_short_selling(symbol: str, qty: float, price: float) -> bool:
    """Validate a proposed short sale.

    Args:
        symbol: The trading symbol.
        qty: Number of shares to short.
        price: Proposed execution price.

    Returns:
        ``True`` if parameters appear valid.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if not symbol:
        raise ValueError("missing_symbol")
    if qty <= 0:
        raise ValueError("invalid_qty")
    if price is not None and price <= 0:
        raise ValueError("invalid_price")
    return True  # Engine expects ``True`` on successful validation


__all__ = [
    "SHORT_SIDE_ALIASES",
    "is_short_side",
    "normalize_short_side",
    "validate_short_selling",
]
