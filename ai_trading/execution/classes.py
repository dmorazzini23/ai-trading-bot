"""Lightweight execution classes used across the trading system.

This module provides minimal implementations of :class:`ExecutionResult` and
``OrderRequest`` that are independent from the heavier production execution
engine.  They are designed to be imported without triggering optional broker
dependencies, making them safe for use in tests and utility scripts.

Both classes leverage ``dataclasses`` to automatically generate ``__init__``
methods and sensible representations.  All public methods return either the
instance itself (to allow chaining) or a structured ``dict`` – avoiding ``None``
returns that complicate caller logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace, fields
from datetime import UTC, datetime
import time
from typing import Any

from ..core.enums import OrderSide, OrderType


@dataclass
class ExecutionResult:
    """Represents the outcome of an order execution."""

    status: str
    order_id: str
    symbol: str
    side: str | None = None
    quantity: int | None = None
    fill_price: float | None = None
    message: str = ""
    execution_time: datetime | None = None
    actual_slippage_bps: float = 0.0
    execution_time_ms: float = 0.0
    notional_value: float = 0.0
    error_code: str | None = None
    venue: str = "simulation"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.execution_time is None:
            self.execution_time = datetime.now(UTC)

    @property
    def is_successful(self) -> bool:
        """Return ``True`` if the execution completed successfully."""

        return self.status.lower() == "success"

    @property
    def is_failed(self) -> bool:
        """Return ``True`` if the execution failed or was rejected."""

        return self.status.lower() in {"failed", "error", "rejected"}

    @property
    def is_partial(self) -> bool:
        """Return ``True`` if the execution partially filled."""

        return self.status.lower() in {"partial", "partially_filled"}

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a structured ``dict``."""

        return {
            "status": self.status,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "message": self.message,
            "execution_time": self.execution_time.isoformat()
            if self.execution_time
            else None,
            "timestamp": self.timestamp.isoformat(),
            "actual_slippage_bps": self.actual_slippage_bps,
            "execution_time_ms": self.execution_time_ms,
            "notional_value": self.notional_value,
            "error_code": self.error_code,
            "venue": self.venue,
            "is_successful": self.is_successful,
            "is_failed": self.is_failed,
            "is_partial": self.is_partial,
        }

    def with_updates(self, **updates: Any) -> "ExecutionResult":
        """Return a copy of the result with ``updates`` applied."""

        return replace(self, **updates)


@dataclass
class OrderRequest:
    """Encapsulates order parameters with basic validation."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: float | None = None
    strategy: str = "unknown"
    time_in_force: str = "DAY"
    client_order_id: str = field(
        default_factory=lambda: f"req_{int(time.time())}"
    )
    stop_price: float | None = None
    target_price: float | None = None
    min_quantity: int = 0
    max_participation_rate: float = 0.1
    urgency_level: str = "normal"
    notes: str = ""
    max_slippage_bps: int = 50
    position_size_limit: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_system: str = "ai_trading"
    request_id: str = field(
        default_factory=lambda: f"req_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    _validation_errors: list[str] = field(default_factory=list, init=False, repr=False)
    _is_valid: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper() if self.symbol else ""
        self._is_valid = self._validate()

    def _validate(self) -> bool:
        self._validation_errors = []
        if not self.symbol:
            self._validation_errors.append("Symbol is required")
        if self.quantity <= 0:
            self._validation_errors.append("Quantity must be positive")
        if self.quantity > 1_000_000:
            self._validation_errors.append("Quantity exceeds maximum limit")
        if not isinstance(self.side, OrderSide):
            self._validation_errors.append("Side must be a valid OrderSide enum")
        if not isinstance(self.order_type, OrderType):
            self._validation_errors.append(
                "Order type must be a valid OrderType enum"
            )
        if self.order_type == OrderType.LIMIT and (self.price is None or self.price <= 0):
            self._validation_errors.append(
                "Limit orders require a valid price"
            )
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and (
            self.stop_price is None or self.stop_price <= 0
        ):
            self._validation_errors.append(
                "Stop orders require a valid stop price"
            )
        if self.max_slippage_bps < 0 or self.max_slippage_bps > 1000:
            self._validation_errors.append(
                "Max slippage must be between 0 and 1000 basis points"
            )
        if self.max_participation_rate <= 0 or self.max_participation_rate > 1:
            self._validation_errors.append(
                "Max participation rate must be between 0 and 1"
            )
        return not self._validation_errors

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def validation_errors(self) -> list[str]:
        return self._validation_errors.copy()

    @property
    def notional_value(self) -> float:
        price = self.price or 100.0
        return abs(self.quantity * price)

    def validate(self) -> tuple[bool, list[str]]:
        self._is_valid = self._validate()
        return self._is_valid, self.validation_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, OrderSide) else self.side,
            "quantity": self.quantity,
            "order_type": self.order_type.value
            if isinstance(self.order_type, OrderType)
            else self.order_type,
            "price": self.price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "time_in_force": self.time_in_force,
            "client_order_id": self.client_order_id,
            "strategy": self.strategy,
            "min_quantity": self.min_quantity,
            "max_participation_rate": self.max_participation_rate,
            "urgency_level": self.urgency_level,
            "max_slippage_bps": self.max_slippage_bps,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "source_system": self.source_system,
            "request_id": self.request_id,
            "notional_value": self.notional_value,
            "is_valid": self.is_valid,
        }

    def to_api_request(self, broker_format: str = "alpaca") -> dict[str, Any]:
        if broker_format.lower() == "alpaca":
            return {
                "symbol": self.symbol,
                "side": self.side.value,
                "type": self.order_type.value,
                "qty": str(self.quantity),
                "time_in_force": self.time_in_force,
                "client_order_id": self.client_order_id,
            }
        return self.to_dict()

    def __repr__(self) -> str:  # pragma: no cover - simple formatting
        field_parts = []
        for f in fields(self):
            if not f.repr:
                continue
            value = getattr(self, f.name)
            if isinstance(value, (OrderSide, OrderType)):
                value = value.value
            field_parts.append(f"{f.name}={value!r}")
        joined = ", ".join(field_parts)
        return f"{self.__class__.__name__}({joined})"

    __str__ = __repr__

    def copy(self, **updates: Any) -> "OrderRequest":
        if "client_order_id" not in updates:
            updates["client_order_id"] = f"req_{int(time.time() * 1000)}"
        return replace(self, **updates)

    def as_validated(self) -> "OrderRequest":
        self._is_valid = self._validate()
        return self

