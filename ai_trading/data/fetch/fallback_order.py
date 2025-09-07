from __future__ import annotations

"""State tracking for provider fallback order."""

from typing import Dict

# Public dictionary tracking fallback usage.
FALLBACK_ORDER: Dict[str, bool] = {}


def mark_yahoo() -> None:
    """Record that Yahoo was used as a fallback."""
    FALLBACK_ORDER["yahoo"] = True


def reset() -> None:
    """Reset tracked state (used in tests)."""
    FALLBACK_ORDER.clear()


__all__ = ["FALLBACK_ORDER", "mark_yahoo", "reset"]
