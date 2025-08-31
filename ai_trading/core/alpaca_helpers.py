from __future__ import annotations

"""Alpaca helper aliases for backwards-compatible imports.

This module re-exports Alpaca-related helpers from ``core.bot_engine``.
It allows gradual module decomposition while keeping the runtime stable.
"""

from .bot_engine import (
    _alpaca_available as alpaca_available,
    list_open_orders,
    ensure_alpaca_attached,
    init_alpaca_clients,
)

__all__ = [
    "alpaca_available",
    "list_open_orders",
    "ensure_alpaca_attached",
    "init_alpaca_clients",
]

