"""Environment variable validation entrypoint."""

from __future__ import annotations

import os
import sys


def _main() -> None:
    # minimal non-verbose validation; do not print secrets
    required = [
        "WEBHOOK_SECRET",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_BASE_URL",
    ]
    for key in required:
        if not os.getenv(key, ""):
            raise RuntimeError(f"Missing required env: {key}")
    if len(os.getenv("WEBHOOK_SECRET", "")) < 32:
        raise RuntimeError("WEBHOOK_SECRET must be >= 32 characters")


__all__ = ["_main"]

