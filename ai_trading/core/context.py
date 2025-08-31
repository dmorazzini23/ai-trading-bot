from __future__ import annotations

"""Context aliases for backwards-compatible imports.

Re-exports selected context-related helpers from bot_engine to allow imports
from a smaller, focused module without breaking existing code.
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

