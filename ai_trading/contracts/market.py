"""Canonical market/execution snapshot contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _normalize_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: datetime | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    timeframe: str | None = None
    provider: str | None = None
    feed: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts.isoformat() if isinstance(self.ts, datetime) else None,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timeframe": self.timeframe,
            "provider": self.provider,
            "feed": self.feed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Bar:
        return cls(
            symbol=str(payload.get("symbol", "") or "").upper(),
            ts=_normalize_timestamp(payload.get("ts") or payload.get("timestamp")),
            open=_safe_float(payload.get("open")),
            high=_safe_float(payload.get("high")),
            low=_safe_float(payload.get("low")),
            close=_safe_float(payload.get("close")),
            volume=_safe_float(payload.get("volume")),
            timeframe=_safe_text(payload.get("timeframe")),
            provider=_safe_text(payload.get("provider")),
            feed=_safe_text(payload.get("feed")),
            metadata=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    ts: datetime | None
    bid: float | None
    ask: float | None
    mid: float | None
    last: float | None
    provider: str | None = None
    feed: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts.isoformat() if isinstance(self.ts, datetime) else None,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "last": self.last,
            "provider": self.provider,
            "feed": self.feed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Quote:
        return cls(
            symbol=str(payload.get("symbol", "") or "").upper(),
            ts=_normalize_timestamp(payload.get("ts") or payload.get("timestamp")),
            bid=_safe_float(payload.get("bid")),
            ask=_safe_float(payload.get("ask")),
            mid=_safe_float(payload.get("mid")),
            last=_safe_float(payload.get("last")),
            provider=_safe_text(payload.get("provider")),
            feed=_safe_text(payload.get("feed")),
            metadata=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    client_order_id: str | None
    broker_order_id: str | None
    side: str | None
    qty: float | None
    filled_qty: float | None
    limit_price: float | None
    fill_price: float | None
    status: str | None
    venue: str | None = None
    ts: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "side": self.side,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "limit_price": self.limit_price,
            "fill_price": self.fill_price,
            "status": self.status,
            "venue": self.venue,
            "ts": self.ts.isoformat() if isinstance(self.ts, datetime) else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BrokerOrderSnapshot:
        return cls(
            client_order_id=_safe_text(payload.get("client_order_id")),
            broker_order_id=_safe_text(
                payload.get("broker_order_id") or payload.get("order_id")
            ),
            side=_safe_text(payload.get("side")),
            qty=_safe_float(payload.get("qty")),
            filled_qty=_safe_float(payload.get("filled_qty")),
            limit_price=_safe_float(payload.get("limit_price") or payload.get("price")),
            fill_price=_safe_float(payload.get("fill_price")),
            status=_safe_text(payload.get("status")),
            venue=_safe_text(payload.get("venue")),
            ts=_normalize_timestamp(payload.get("ts") or payload.get("timestamp")),
            metadata=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    submitted: bool
    accepted: bool
    status: str | None
    provider: str | None = None
    venue: str | None = None
    broker_order: BrokerOrderSnapshot | None = None
    fill_count: int = 0
    filled_qty: float | None = None
    realized_slippage_bps: float | None = None
    fees: float | None = None
    error_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "accepted": self.accepted,
            "status": self.status,
            "provider": self.provider,
            "venue": self.venue,
            "broker_order": (
                self.broker_order.to_dict() if self.broker_order is not None else None
            ),
            "fill_count": self.fill_count,
            "filled_qty": self.filled_qty,
            "realized_slippage_bps": self.realized_slippage_bps,
            "fees": self.fees,
            "error_reason": self.error_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ExecutionResult:
        broker_order_payload = payload.get("broker_order")
        broker_order = (
            BrokerOrderSnapshot.from_mapping(broker_order_payload)
            if isinstance(broker_order_payload, Mapping)
            else None
        )
        return cls(
            submitted=_safe_bool(payload.get("submitted")),
            accepted=_safe_bool(payload.get("accepted")),
            status=_safe_text(payload.get("status")),
            provider=_safe_text(payload.get("provider")),
            venue=_safe_text(payload.get("venue")),
            broker_order=broker_order,
            fill_count=_safe_int(payload.get("fill_count"), default=0),
            filled_qty=_safe_float(payload.get("filled_qty")),
            realized_slippage_bps=_safe_float(payload.get("realized_slippage_bps")),
            fees=_safe_float(payload.get("fees")),
            error_reason=_safe_text(payload.get("error_reason")),
            metadata=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    symbol: str
    qty: float
    market_value: float | None = None
    avg_entry_price: float | None = None
    provider: str | None = None
    ts: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "market_value": self.market_value,
            "avg_entry_price": self.avg_entry_price,
            "provider": self.provider,
            "ts": self.ts.isoformat() if isinstance(self.ts, datetime) else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PositionSnapshot:
        return cls(
            symbol=str(payload.get("symbol", "") or "").upper(),
            qty=float(payload.get("qty") or 0.0),
            market_value=_safe_float(payload.get("market_value")),
            avg_entry_price=_safe_float(payload.get("avg_entry_price")),
            provider=_safe_text(payload.get("provider")),
            ts=_normalize_timestamp(payload.get("ts") or payload.get("timestamp")),
            metadata=dict(payload),
        )


__all__ = [
    "Bar",
    "BrokerOrderSnapshot",
    "ExecutionResult",
    "PositionSnapshot",
    "Quote",
]
