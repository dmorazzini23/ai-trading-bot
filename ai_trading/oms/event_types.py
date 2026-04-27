"""Typed OMS and decision event payload contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
import uuid

OmsEventType = Literal[
    "DECISION_EMITTED",
    "INTENT_CREATED",
    "SUBMIT_CLAIMED",
    "SUBMIT_ATTEMPTED",
    "SUBMIT_ACK",
    "SUBMIT_REJECT",
    "ORDER_PARTIALLY_FILLED",
    "ORDER_FILLED",
    "ORDER_CANCELED",
    "ORDER_FAILED",
    "INTENT_CLOSED",
    "RECONCILE_UPDATE",
]

DecisionAction = Literal["BUY", "SELL", "SELL_SHORT", "HOLD", "REDUCE", "EXIT"]


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601."""

    return datetime.now(UTC).isoformat()


def new_event_uuid() -> str:
    """Return random event UUID string."""

    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class OmsEvent:
    """Immutable OMS lifecycle event."""

    event_type: OmsEventType
    event_source: str
    idempotency_key: str
    payload: dict[str, Any]
    intent_id: str | None = None
    sequence_no: int | None = None
    event_ts: str | None = None
    event_uuid: str | None = None
    policy_hash: str | None = None
    model_hash: str | None = None
    error_code: str | None = None
    broker_order_id: str | None = None
    fill_id: str | None = None

    def normalized(self) -> "OmsEvent":
        """Return a normalized immutable event."""

        normalized_ts = str(self.event_ts or utc_now_iso())
        normalized_uuid = str(self.event_uuid or new_event_uuid())
        normalized_source = str(self.event_source or "").strip() or "unknown"
        normalized_key = str(self.idempotency_key or "").strip()
        if not normalized_key:
            raise ValueError("idempotency_key is required for OmsEvent")
        return OmsEvent(
            event_type=self.event_type,
            event_source=normalized_source,
            idempotency_key=normalized_key,
            payload=dict(self.payload or {}),
            intent_id=(str(self.intent_id) if self.intent_id not in (None, "") else None),
            sequence_no=self.sequence_no,
            event_ts=normalized_ts,
            event_uuid=normalized_uuid,
            policy_hash=(str(self.policy_hash) if self.policy_hash not in (None, "") else None),
            model_hash=(str(self.model_hash) if self.model_hash not in (None, "") else None),
            error_code=(str(self.error_code) if self.error_code not in (None, "") else None),
            broker_order_id=(
                str(self.broker_order_id) if self.broker_order_id not in (None, "") else None
            ),
            fill_id=(str(self.fill_id) if self.fill_id not in (None, "") else None),
        )


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """Immutable decision event for audit lineage."""

    symbol: str
    decision_action: DecisionAction
    decision_source: str
    idempotency_key: str
    strategy_id: str | None = None
    confidence: float | None = None
    expected_edge_bps: float | None = None
    policy_hash: str | None = None
    model_hash: str | None = None
    config_hash: str | None = None
    features: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    decision_uuid: str | None = None
    decision_ts: str | None = None

    def normalized(self) -> "DecisionEvent":
        """Return a normalized immutable decision event."""

        normalized_symbol = str(self.symbol or "").strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol is required for DecisionEvent")
        normalized_key = str(self.idempotency_key or "").strip()
        if not normalized_key:
            raise ValueError("idempotency_key is required for DecisionEvent")
        normalized_source = str(self.decision_source or "").strip() or "unknown"
        normalized_ts = str(self.decision_ts or utc_now_iso())
        normalized_uuid = str(self.decision_uuid or new_event_uuid())
        return DecisionEvent(
            symbol=normalized_symbol,
            decision_action=self.decision_action,
            decision_source=normalized_source,
            idempotency_key=normalized_key,
            strategy_id=(str(self.strategy_id) if self.strategy_id not in (None, "") else None),
            confidence=(float(self.confidence) if self.confidence is not None else None),
            expected_edge_bps=(
                float(self.expected_edge_bps) if self.expected_edge_bps is not None else None
            ),
            policy_hash=(str(self.policy_hash) if self.policy_hash not in (None, "") else None),
            model_hash=(str(self.model_hash) if self.model_hash not in (None, "") else None),
            config_hash=(str(self.config_hash) if self.config_hash not in (None, "") else None),
            features=dict(self.features or {}),
            context=dict(self.context or {}),
            decision_uuid=normalized_uuid,
            decision_ts=normalized_ts,
        )
