from __future__ import annotations

"""Context aliases for backwards-compatible imports.

This module re-exports context-related symbols from ``core.bot_engine`` so
callers can begin importing from a smaller, focused module without breaking
existing code. The underlying implementation continues to live in
``bot_engine`` to avoid any runtime risk during market hours.
"""

from .bot_engine import (
    BotContext,
    LazyBotContext,
    get_ctx,
    ensure_alpaca_attached,
    maybe_init_brokers,
    init_alpaca_clients,
)

__all__ = [
    "BotContext",
    "LazyBotContext",
    "get_ctx",
    "ensure_alpaca_attached",
    "maybe_init_brokers",
    "init_alpaca_clients",
]

