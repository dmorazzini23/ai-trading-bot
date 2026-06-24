"""OMS reconciliation invariants over durable intents and immutable events."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from typing import Any

from ai_trading.logging import get_logger

from .lifecycle import terminal_event_type
from .statuses import is_terminal_intent_status, normalize_intent_status

logger = get_logger(__name__)


def evaluate_oms_reconciliation_invariants(
    *,
    database_url: str | None = None,
    intent_store_path: str | None = None,
    limit: int = 5000,
) -> dict[str, Any]:
    """Evaluate event/intents consistency invariants for runtime go/no-go checks."""

    try:
        from ai_trading.oms.event_store import EventStore
        from ai_trading.oms.intent_store import IntentStore
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {"available": False, "ok": False, "error": str(exc)}

    intent_store = IntentStore(path=intent_store_path, url=database_url)
    event_store = EventStore(path=intent_store_path, url=database_url)
    violations = {
        "missing_intent_created": 0,
        "submitted_missing_ack": 0,
        "terminal_missing_close": 0,
        "filled_missing_fill_event": 0,
    }
    samples: list[dict[str, Any]] = []
    scanned = 0

    def _record(issue: str, *, intent_id: str, status: str, event_types: set[str]) -> None:
        if issue in violations:
            violations[issue] += 1
        if len(samples) < 25:
            samples.append(
                {
                    "issue": issue,
                    "intent_id": intent_id,
                    "status": status,
                    "event_types": sorted(event_types),
                }
            )

    try:
        intents = intent_store.list_intents(limit=max(1, int(limit)))
        for intent in intents:
            scanned += 1
            events = event_store.list_oms_events(intent_id=intent.intent_id, limit=5000)
            event_types = {str(row.get("event_type") or "").strip().upper() for row in events}
            event_types.discard("")

            if "INTENT_CREATED" not in event_types:
                _record(
                    "missing_intent_created",
                    intent_id=intent.intent_id,
                    status=str(intent.status),
                    event_types=event_types,
                )
            if str(intent.broker_order_id or "").strip() and "SUBMIT_ACK" not in event_types:
                _record(
                    "submitted_missing_ack",
                    intent_id=intent.intent_id,
                    status=str(intent.status),
                    event_types=event_types,
                )

            normalized_status = normalize_intent_status(intent.status, default="PENDING_SUBMIT")
            if is_terminal_intent_status(normalized_status) and "INTENT_CLOSED" not in event_types:
                _record(
                    "terminal_missing_close",
                    intent_id=intent.intent_id,
                    status=normalized_status,
                    event_types=event_types,
                )
            if normalized_status == "FILLED" and not (
                {"ORDER_FILLED", "ORDER_PARTIALLY_FILLED"} & event_types
            ):
                _record(
                    "filled_missing_fill_event",
                    intent_id=intent.intent_id,
                    status=normalized_status,
                    event_types=event_types,
                )
    finally:
        try:
            intent_store.close()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("OMS_INVARIANTS_INTENT_STORE_CLOSE_FAILED", exc_info=True)
        try:
            event_store.close()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("OMS_INVARIANTS_EVENT_STORE_CLOSE_FAILED", exc_info=True)

    total_violations = int(sum(int(value) for value in violations.values()))
    return {
        "available": True,
        "ok": total_violations == 0,
        "scanned_intents": int(scanned),
        "violations": dict(violations),
        "total_violations": total_violations,
        "sample_violations": list(samples),
    }


def evaluate_oms_lifecycle_parity_invariants(
    *,
    database_url: str | None = None,
    intent_store_path: str | None = None,
    limit: int = 5000,
) -> dict[str, Any]:
    """Evaluate strict lifecycle parity invariants shared by live and backtest paths."""

    try:
        from ai_trading.oms.event_store import EventStore
        from ai_trading.oms.intent_store import IntentStore
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        return {"available": False, "ok": False, "error": str(exc)}

    intent_store = IntentStore(path=intent_store_path, url=database_url)
    event_store = EventStore(path=intent_store_path, url=database_url)
    violations = {
        "missing_submit_claim": 0,
        "missing_submit_attempt": 0,
        "missing_submit_ack": 0,
        "lifecycle_ordering_invalid": 0,
        "terminal_event_mismatch": 0,
        "terminal_missing_close": 0,
        "filled_missing_partial_fill": 0,
    }
    samples: list[dict[str, Any]] = []
    scanned = 0

    def _record(
        issue: str,
        *,
        intent_id: str,
        status: str,
        event_types: list[str],
    ) -> None:
        if issue in violations:
            violations[issue] += 1
        if len(samples) < 25:
            samples.append(
                {
                    "issue": issue,
                    "intent_id": intent_id,
                    "status": status,
                    "event_types": list(event_types),
                }
            )

    def _first_index(haystack: list[str], needle: str) -> int | None:
        try:
            return haystack.index(needle)
        except ValueError:
            return None

    def _terminal_cancel_without_submit_ack_allowed(
        *,
        normalized_status: str,
        broker_order_id: str,
        event_set: set[str],
    ) -> bool:
        if normalized_status not in {"CANCELED", "CANCELLED", "EXPIRED", "DONE_FOR_DAY"}:
            return False
        if str(broker_order_id or "").strip():
            return False
        expected_terminal_event = terminal_event_type(normalized_status)
        return bool(
            expected_terminal_event in event_set
            and "INTENT_CLOSED" in event_set
            and "SUBMIT_ACK" not in event_set
        )

    try:
        intents = intent_store.list_intents(limit=max(1, int(limit)))
        for intent in intents:
            scanned += 1
            rows = event_store.list_oms_events(intent_id=intent.intent_id, limit=5000)
            event_types = [
                str(row.get("event_type") or "").strip().upper()
                for row in rows
                if str(row.get("event_type") or "").strip()
            ]
            event_set = set(event_types)
            normalized_status = normalize_intent_status(
                intent.status,
                default="PENDING_SUBMIT",
            )
            requires_submit_path = (
                normalized_status != "PENDING_SUBMIT"
                or str(intent.broker_order_id or "").strip() != ""
                or "SUBMIT_ACK" in event_set
            )
            if requires_submit_path and "SUBMIT_CLAIMED" not in event_set:
                _record(
                    "missing_submit_claim",
                    intent_id=intent.intent_id,
                    status=normalized_status,
                    event_types=event_types,
                )
            if requires_submit_path and "SUBMIT_ATTEMPTED" not in event_set:
                _record(
                    "missing_submit_attempt",
                    intent_id=intent.intent_id,
                    status=normalized_status,
                    event_types=event_types,
                )
            if requires_submit_path and "SUBMIT_ACK" not in event_set:
                cancel_without_ack_allowed = _terminal_cancel_without_submit_ack_allowed(
                    normalized_status=normalized_status,
                    broker_order_id=str(intent.broker_order_id or ""),
                    event_set=event_set,
                )
                if normalized_status not in {"FAILED", "REJECTED"} and not cancel_without_ack_allowed:
                    _record(
                        "missing_submit_ack",
                        intent_id=intent.intent_id,
                        status=normalized_status,
                        event_types=event_types,
                    )

            created_idx = _first_index(event_types, "INTENT_CREATED")
            claimed_idx = _first_index(event_types, "SUBMIT_CLAIMED")
            attempted_idx = _first_index(event_types, "SUBMIT_ATTEMPTED")
            ack_idx = _first_index(event_types, "SUBMIT_ACK")
            if (
                created_idx is not None
                and claimed_idx is not None
                and attempted_idx is not None
                and ack_idx is not None
                and not (
                    created_idx <= claimed_idx <= attempted_idx <= ack_idx
                )
            ):
                _record(
                    "lifecycle_ordering_invalid",
                    intent_id=intent.intent_id,
                    status=normalized_status,
                    event_types=event_types,
                )

            if is_terminal_intent_status(normalized_status):
                expected_terminal_event = terminal_event_type(normalized_status)
                if (
                    expected_terminal_event != "INTENT_CLOSED"
                    and expected_terminal_event not in event_set
                ):
                    _record(
                        "terminal_event_mismatch",
                        intent_id=intent.intent_id,
                        status=normalized_status,
                        event_types=event_types,
                    )
                if "INTENT_CLOSED" not in event_set:
                    _record(
                        "terminal_missing_close",
                        intent_id=intent.intent_id,
                        status=normalized_status,
                        event_types=event_types,
                    )
            if normalized_status == "FILLED":
                order_filled_count = sum(1 for event_type in event_types if event_type == "ORDER_FILLED")
                partial_fill_count = sum(
                    1 for event_type in event_types if event_type == "ORDER_PARTIALLY_FILLED"
                )
                if partial_fill_count == 0 and order_filled_count < 2:
                    _record(
                        "filled_missing_partial_fill",
                        intent_id=intent.intent_id,
                        status=normalized_status,
                        event_types=event_types,
                    )
    finally:
        try:
            intent_store.close()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("OMS_PARITY_INTENT_STORE_CLOSE_FAILED", exc_info=True)
        try:
            event_store.close()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("OMS_PARITY_EVENT_STORE_CLOSE_FAILED", exc_info=True)

    total_violations = int(sum(int(value) for value in violations.values()))
    return {
        "available": True,
        "ok": total_violations == 0,
        "scanned_intents": int(scanned),
        "violations": dict(violations),
        "total_violations": total_violations,
        "sample_violations": list(samples),
    }


__all__ = [
    "evaluate_oms_reconciliation_invariants",
    "evaluate_oms_lifecycle_parity_invariants",
]
