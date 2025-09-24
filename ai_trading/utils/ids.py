"""Utility helpers for generating stable client identifiers."""

from __future__ import annotations

import secrets

__all__ = ["stable_client_order_id"]


def stable_client_order_id(symbol: str, side: str, epoch_min: int) -> str:
    """Return a deterministic client order id prefix with a random suffix."""

    suffix = secrets.token_hex(4)
    base = f"{symbol}-{side}-{epoch_min}"
    return f"{base}-{suffix}"
