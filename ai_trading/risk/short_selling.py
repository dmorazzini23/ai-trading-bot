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


def borrowable_share_count(asset: Any) -> int | None:
    raw = _read_attr(
        asset,
        "shortable_shares",
        "easy_to_borrow_shares",
        "borrowable_shares",
        "available_borrow_shares",
    )
    if raw is None:
        return None
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return None


def is_asset_borrowable(asset: Any, *, qty: float | None = None) -> bool:
    shortable = _bool_from_record(asset, "shortable", "is_shortable", "shortable_flag")
    if shortable is False:
        return False
    easy = _bool_from_record(
        asset,
        "easy_to_borrow",
        "easy_to_borrow_flag",
        "easy_to_borrow_shares",
        "borrowable",
    )
    if easy is False:
        return False
    available = borrowable_share_count(asset)
    if qty is not None and available is not None:
        try:
            requested = float(qty)
        except (TypeError, ValueError):
            return False
        if requested > available:
            return False
    return shortable is True or easy is True or (available is not None and available > 0)


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
    "borrowable_share_count",
    "is_asset_borrowable",
    "is_short_side",
    "normalize_short_side",
    "validate_short_selling",
]
