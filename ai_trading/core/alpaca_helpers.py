from __future__ import annotations

"""Alpaca helper aliases for backwards-compatible imports."""

from ai_trading.core.alpaca_client import (
    list_open_orders,
    ensure_alpaca_attached,
    _initialize_alpaca_clients as init_alpaca_clients,
)
from ai_trading.core.bot_engine import _alpaca_available as alpaca_available

__all__ = [
    "alpaca_available",
    "list_open_orders",
    "ensure_alpaca_attached",
    "init_alpaca_clients",
]

