from __future__ import annotations

import os


def _main() -> None:
    """Validate required environment variables for runtime."""
    webhook = os.getenv("WEBHOOK_SECRET", "")
    if len(webhook) < 32:
        raise RuntimeError("WEBHOOK_SECRET too short")
    api = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    base = os.getenv("ALPACA_BASE_URL", "")
    if not (api and secret and base):
        raise RuntimeError("Missing alpaca environment variables")


__all__ = ["_main"]
