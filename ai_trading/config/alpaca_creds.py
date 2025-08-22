from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def resolve_alpaca_credentials():
    """Resolve Alpaca credentials using only ALPACA_* variables."""  # AI-AGENT-REF: drop legacy aliases
    out = {
        "API_KEY": os.getenv("ALPACA_API_KEY"),
        "SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
        "BASE_URL": os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets",
    }
    return out


__all__ = ["resolve_alpaca_credentials"]

