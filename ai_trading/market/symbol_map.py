from __future__ import annotations

"""Utility for broker-specific symbol normalization."""

_ALPACA_OVERRIDES = {"BRK-B": "BRK.B"}


def to_alpaca_symbol(symbol: str) -> str:
    """Return symbol normalized for Alpaca REST calls."""
    s = symbol.strip().upper()
    return _ALPACA_OVERRIDES.get(s, s)
