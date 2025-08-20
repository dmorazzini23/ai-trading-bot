from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def resolve_alpaca_credentials():
    """
    Resolution rules:
      1) ALPACA_* takes precedence over APCA_* for each field.
      2) If both set and different, log a warning.
      3) Allow partial sets (None allowed); BASE_URL defaults to paper if missing.
    """
    pairs = [
        ("API_KEY", "ALPACA_API_KEY", "APCA_API_KEY_ID"),
        ("SECRET_KEY", "ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"),
        ("BASE_URL", "ALPACA_BASE_URL", "APCA_API_BASE_URL"),
    ]
    out = {}
    for label, prefer, alt in pairs:
        p = os.getenv(prefer)
        a = os.getenv(alt)
        if p and a and p != a:
            log.warning("Conflicting %s: %s vs %s; preferring %s", label, prefer, alt, prefer)
        out[label] = p or a
    if not out.get("BASE_URL"):
        out["BASE_URL"] = "https://paper-api.alpaca.markets"
    return out


__all__ = ["resolve_alpaca_credentials"]

