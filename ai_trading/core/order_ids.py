from __future__ import annotations

"""Utilities for generating client order IDs.

This module wraps the Alpaca helper so it can be easily patched during tests.
"""

from ai_trading.alpaca_api import (
    generate_client_order_id as _generate_client_order_id,
)


def generate_client_order_id(prefix: str = "ai") -> str:
    """Return a unique client order ID with the given prefix."""
    return _generate_client_order_id(prefix)
