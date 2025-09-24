"""Utilities for generating client order IDs.

This module wraps the Alpaca helper so it can be easily patched during tests
and also exposes deterministic helpers used for retry idempotency.
"""

from __future__ import annotations

import hashlib
import time

from ai_trading.alpaca_api import (
    generate_client_order_id as _generate_client_order_id,
)


def generate_client_order_id(prefix: str = "ai") -> str:
    """Return a unique client order ID with the given prefix."""
    return _generate_client_order_id(prefix)


def stable_client_order_id(symbol: str, side: str, epoch_min: int | None = None) -> str:
    """Return a deterministic client order identifier.

    Parameters
    ----------
    symbol:
        Trading symbol for the order.
    side:
        Order side (``"buy"``/``"sell"``). Case is normalized to lower-case.
    epoch_min:
        Unix epoch minute used to keep identifiers stable during retries. When
        omitted, the current minute is used.
    """

    epoch_source = int(epoch_min if epoch_min is not None else time.time() // 60)
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_side = str(side or "").strip().lower()
    payload = f"{normalized_symbol}:{normalized_side}:{epoch_source}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:18]
    return f"ai-{digest}"
