"""Short selling validation utilities."""
from __future__ import annotations

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
    return True
