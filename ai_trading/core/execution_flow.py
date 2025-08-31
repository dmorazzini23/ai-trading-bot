from __future__ import annotations

"""Execution flow aliases for backwards-compatible imports.

Exports execution-related primitives from ``core.bot_engine`` so new call
sites can import from a focused module without changing behavior.
"""

from .bot_engine import (
    submit_order,
    safe_submit_order,
    poll_order_fill_status,
    execute_exit,
)

__all__ = [
    "submit_order",
    "safe_submit_order",
    "poll_order_fill_status",
    "execute_exit",
]

