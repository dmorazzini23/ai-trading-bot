"""Shared OMS lifecycle/status transition helpers."""

from __future__ import annotations

from ai_trading.oms.statuses import (
    TERMINAL_INTENT_STATUSES,
    is_terminal_intent_status,
    normalize_intent_status,
)

PENDING_SUBMIT_STATUS = "PENDING_SUBMIT"
SUBMITTING_STATUS = "SUBMITTING"
SUBMITTED_STATUS = "SUBMITTED"
PARTIALLY_FILLED_STATUS = "PARTIALLY_FILLED"

SUBMIT_CLAIMABLE_STATUSES: frozenset[str] = frozenset(
    {PENDING_SUBMIT_STATUS, SUBMITTING_STATUS}
)

_PRESERVE_FILL_STATUSES: frozenset[str] = frozenset({"FILLED", "CLOSED"})

_TERMINAL_EVENT_BY_STATUS: dict[str, str] = {
    "FILLED": "ORDER_FILLED",
    "CANCELED": "ORDER_CANCELED",
    "CANCELLED": "ORDER_CANCELED",
    "EXPIRED": "ORDER_CANCELED",
    "DONE_FOR_DAY": "ORDER_CANCELED",
    "FAILED": "ORDER_FAILED",
    "REJECTED": "ORDER_FAILED",
}

_BROKER_TERMINAL_STATUS_MAP: dict[str, str] = {
    "DONE_FOR_DAY": "EXPIRED",
    "REPLACED": "CLOSED",
    "STOPPED": "CLOSED",
    "STOPPED_OUT": "CLOSED",
}

_BROKER_TERMINAL_TOKENS: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "CANCELLED",
        "CANCELED",
        "EXPIRED",
        "DONE_FOR_DAY",
        "REPLACED",
        "STOPPED",
        "STOPPED_OUT",
        "FAILED",
        "REJECTED",
        "FILLED",
    }
)


def _normalize_token(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    token = str(value).strip().upper()
    if "." in token:
        token = token.split(".")[-1]
    if token in {"NONE", "NULL", "N/A", "NA"}:
        return None
    return token or None


def status_for_submit_claim() -> str:
    return SUBMITTING_STATUS


def status_for_submit_ack() -> str:
    return SUBMITTED_STATUS


def status_for_submit_error() -> str:
    return PENDING_SUBMIT_STATUS


def status_for_fill(current_status: str | None) -> str:
    normalized = normalize_intent_status(current_status, default=PARTIALLY_FILLED_STATUS)
    if normalized in _PRESERVE_FILL_STATUSES:
        return str(normalized)
    return PARTIALLY_FILLED_STATUS


def normalize_terminal_status(status: str | None) -> str:
    normalized = normalize_intent_status(status, default="CLOSED")
    if not is_terminal_intent_status(normalized):
        raise ValueError(f"close_intent requires terminal status, got: {normalized}")
    return str(normalized)


def terminal_event_type(status: str | None) -> str:
    normalized = normalize_intent_status(status, default="CLOSED")
    return _TERMINAL_EVENT_BY_STATUS.get(normalized, "INTENT_CLOSED")


def resolve_terminal_intent_status(
    *,
    status: str | None = None,
    event_type: str | None = None,
    status_is_terminal: bool = False,
) -> str | None:
    """Resolve broker state into canonical terminal intent status."""

    status_token = _normalize_token(status)
    event_token = _normalize_token(event_type)
    candidate = status_token or event_token
    mapped = _BROKER_TERMINAL_STATUS_MAP.get(candidate or "", candidate or "")
    is_terminal = (
        bool(status_is_terminal)
        or (status_token in _BROKER_TERMINAL_TOKENS if status_token else False)
        or (event_token in _BROKER_TERMINAL_TOKENS if event_token else False)
        or (mapped in TERMINAL_INTENT_STATUSES if mapped else False)
    )
    if not is_terminal:
        return None
    normalized = normalize_intent_status(mapped, default="CLOSED")
    if not is_terminal_intent_status(normalized):
        return "CLOSED"
    return str(normalized)


__all__ = [
    "PENDING_SUBMIT_STATUS",
    "SUBMITTING_STATUS",
    "SUBMITTED_STATUS",
    "PARTIALLY_FILLED_STATUS",
    "SUBMIT_CLAIMABLE_STATUSES",
    "status_for_submit_claim",
    "status_for_submit_ack",
    "status_for_submit_error",
    "status_for_fill",
    "normalize_terminal_status",
    "terminal_event_type",
    "resolve_terminal_intent_status",
]
