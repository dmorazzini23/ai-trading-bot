"""Canonical OMS intent status helpers."""

from __future__ import annotations

TERMINAL_INTENT_STATUSES: frozenset[str] = frozenset(
    {
        "FILLED",
        "CANCELED",
        "CANCELLED",
        "REJECTED",
        "CLOSED",
        "FAILED",
        "EXPIRED",
        "COMPLETED",
        "DONE_FOR_DAY",
        "REPLACED",
        "STOPPED",
    }
)


def normalize_intent_status(status: str | None, *, default: str = "CLOSED") -> str:
    """Return upper-cased intent status with a fallback default."""

    normalized = str(status).strip().upper() if status is not None else ""
    if not normalized:
        normalized = str(default).strip().upper() or "CLOSED"
    return normalized


def is_terminal_intent_status(status: str | None) -> bool:
    """Return ``True`` when ``status`` is terminal for OMS intents."""

    return normalize_intent_status(status) in TERMINAL_INTENT_STATUSES


__all__ = [
    "TERMINAL_INTENT_STATUSES",
    "normalize_intent_status",
    "is_terminal_intent_status",
]
