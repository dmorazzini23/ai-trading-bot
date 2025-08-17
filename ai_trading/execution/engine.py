"""
Core execution engine for institutional order management.

Provides order lifecycle management, execution algorithms,
and real-time execution monitoring with institutional controls.
"""

# ruff: noqa
from __future__ import annotations

import importlib
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)


def _optional_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# AI-AGENT-REF: replace ImportError guard with optional import
_alpaca_rest = _optional_import("alpaca_trade_api.rest")
if _alpaca_rest is not None:
    APIError = _alpaca_rest.APIError  # type: ignore[attr-defined]
else:  # pragma: no cover - fallback for dev/test

    class APIError(Exception):
        pass


ORDER_STALE_TIMEOUT_S = int(os.getenv("ORDER_STALE_TIMEOUT_S", "300") or 300)


@dataclass
class OrderInfo:
    order_id: str
    symbol: str
    side: str
    qty: int
    submitted_time: float
    last_status: str = "new"


_active_orders: dict[str, OrderInfo] = {}
_order_tracking_lock = threading.Lock()


def _cleanup_stale_orders(now_s: float | None = None) -> list[str]:
    now = now_s if now_s is not None else time.time()
    removed: list[str] = []
    with _order_tracking_lock:
        for oid, info in list(_active_orders.items()):
            age = now - info.submitted_time
            if age >= ORDER_STALE_TIMEOUT_S and info.last_status in {"new", "pending"}:
                removed.append(oid)
                _active_orders.pop(oid, None)
    return removed


from ai_trading.market.symbol_specs import TICK_BY_SYMBOL, get_lot_size, get_tick_size
from ai_trading.math.money import Money, round_to_lot, round_to_tick

from ..core.constants import EXECUTION_PARAMETERS
from ..core.enums import OrderSide, OrderStatus, OrderType
from .idempotency import OrderIdempotencyCache


# AI-AGENT-REF: quantity input guard
def _ensure_positive_qty(qty: float) -> float:
    if qty is None:
        raise ValueError("qty_none")
    q = float(qty)
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError(f"invalid_qty:{qty}")
    return q


# AI-AGENT-REF: price input guard
def _ensure_valid_price(price: float | None) -> float | None:
    if price is None:
        return None
    p = float(price)
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError(f"invalid_price:{price}")
    return p


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""

    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ICEBERG = "iceberg"


class Order:
    """
    Order representation for institutional execution.

    Comprehensive order model with execution tracking,
    partial fills, and institutional metadata.
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Money = None,
        **kwargs,
    ):
        """Initialize order with institutional parameters."""
        # AI-AGENT-REF: Institutional order model with Money precision
        self.id = kwargs.get("id", str(uuid.uuid4()))
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type

        # Use Money for precise price calculations
        tick = TICK_BY_SYMBOL.get(symbol)
        self.price = Money(price, tick) if price is not None else None
        self.status = OrderStatus.PENDING

        # Execution details
        self.filled_quantity = 0
        self.average_fill_price = Money(0)
        self.fills = []
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self.executed_at = None

        # Institutional parameters
        self.client_order_id = kwargs.get("client_order_id", f"ord_{int(time.time())}")
        self.strategy_id = kwargs.get("strategy_id")
        self.execution_algorithm = kwargs.get("execution_algorithm", ExecutionAlgorithm.MARKET)
        self.time_in_force = kwargs.get("time_in_force", "DAY")
        self.min_quantity = kwargs.get("min_quantity", 0)
        self.stop_price = kwargs.get("stop_price")
        self.target_price = kwargs.get("target_price")

        # Risk and compliance
        self.max_participation_rate = kwargs.get("max_participation_rate", 0.1)
        self.max_slippage_bps = kwargs.get(
            "max_slippage_bps", EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"]
        )
        self.urgency_level = kwargs.get("urgency_level", "normal")  # low, normal, high

        # Metadata
        self.notes = kwargs.get("notes", "")
        self.source_system = kwargs.get("source_system", "ai_trading")
        self.parent_order_id = kwargs.get("parent_order_id")

        _log.debug(f"Order created: {self.id} {self.side} {self.quantity} {self.symbol}")

    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity

    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return 0 < self.filled_quantity < self.quantity

    @property
    def notional_value(self) -> Money:
        """Calculate notional value of order with precise money math."""
        price = self.price or self.average_fill_price or Money(0)
        return Money(abs(self.quantity)) * price

    def add_fill(self, quantity: int, price: Money, timestamp: datetime = None):
        """Add a fill to the order with precise money math."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Ensure price uses proper tick size for this symbol
        tick = TICK_BY_SYMBOL.get(self.symbol)
        if not isinstance(price, Money):
            price = Money(price, tick)

        fill = {
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "fill_id": str(uuid.uuid4()),
        }

        self.fills.append(fill)
        self.filled_quantity += quantity

        # Update average fill price with precise math
        total_value = sum(Money(f["quantity"]) * f["price"] for f in self.fills)
        self.average_fill_price = (
            total_value / Money(self.filled_quantity) if self.filled_quantity > 0 else Money(0)
        )

        # Update status
        if self.is_filled:
            self.status = OrderStatus.FILLED
            self.executed_at = timestamp
        elif self.is_partially_filled:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.updated_at = timestamp

        _log.debug(
            f"Fill added to order {self.id}: {quantity}@{price} "
            f"({self.fill_percentage:.1f}% filled)"
        )

    def cancel(self, reason: str = "User cancelled"):
        """Cancel the order."""
        if self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        ]:
            _log.warning(f"Cannot cancel order {self.id} in status {self.status}")
            return False

        self.status = OrderStatus.CANCELED
        self.updated_at = datetime.now(UTC)
        self.notes += f" | Cancelled: {reason}"

        _log.info(f"Order {self.id} cancelled: {reason}")
        return True

    def to_dict(self) -> dict:
        """Convert order to dictionary representation."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, OrderSide) else self.side,
            "quantity": self.quantity,
            "order_type": (
                self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type
            ),
            "price": self.price,
            "status": (self.status.value if isinstance(self.status, OrderStatus) else self.status),
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "client_order_id": self.client_order_id,
            "strategy_id": self.strategy_id,
            "fills": self.fills,
            "notional_value": self.notional_value,
            "fill_percentage": self.fill_percentage,
        }


class OrderManager:
    """
    Order lifecycle management for institutional execution.

    Manages order routing, execution tracking, and provides
    real-time order monitoring with institutional controls.
    """

    def __init__(self):
        """Initialize order manager."""
        # AI-AGENT-REF: Institutional order lifecycle management
        self.orders: dict[str, Order] = {}
        self.active_orders: dict[str, Order] = {}
        self.execution_callbacks: list[Callable] = []

        # Execution parameters
        self.max_concurrent_orders = EXECUTION_PARAMETERS.get("MAX_CONCURRENT_ORDERS", 100)
        self.order_timeout = EXECUTION_PARAMETERS.get("ORDER_TIMEOUT_SECONDS", 300)
        self.retry_attempts = EXECUTION_PARAMETERS.get("RETRY_ATTEMPTS", 3)

        # Threading for order monitoring
        self._monitor_thread = None
        self._monitor_running = False

        self._idempotency_cache: OrderIdempotencyCache | None = None

        _log.info("OrderManager initialized")

    def _ensure_idempotency_cache(self) -> OrderIdempotencyCache:
        """Ensure idempotency cache is instantiated."""
        if self._idempotency_cache is None:
            try:
                # AI-AGENT-REF: instantiate idempotency cache
                self._idempotency_cache = OrderIdempotencyCache()
            except (
                KeyError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                _log.error(
                    "IDEMPOTENCY_CACHE_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )  # AI-AGENT-REF: narrow idempotency cache errors
                raise
        return self._idempotency_cache

    def submit_order(self, order: Order) -> bool:
        """
        Submit order for execution.

        Args:
            order: Order to submit

        Returns:
            True if order was successfully submitted
        """
        try:
            # Validate order
            if not self._validate_order(order):
                return False

            # AI-AGENT-REF: Wire idempotency checking before submission
            cache = self._ensure_idempotency_cache()

            # Generate idempotency key
            key = cache.generate_key(order.symbol, order.side, order.quantity, datetime.now(UTC))

            # Check if this is a duplicate order
            if cache.is_duplicate(key):
                _log.warning(
                    f"ORDER_DUPLICATE_SKIPPED: {order.symbol} {order.side} {order.quantity}"
                )
                order.status = OrderStatus.REJECTED
                order.notes += " | Rejected: Duplicate order detected"
                return False

            # Check capacity
            if len(self.active_orders) >= self.max_concurrent_orders:
                _log.error(
                    f"Cannot submit order: max concurrent orders reached ({self.max_concurrent_orders})"
                )
                order.status = OrderStatus.REJECTED
                order.notes += " | Rejected: Max concurrent orders reached"
                return False

            # Add to tracking
            self.orders[order.id] = order
            self.active_orders[order.id] = order

            # AI-AGENT-REF: Mark order as submitted in idempotency cache
            cache.mark_submitted(key, order.id)

            # Start monitoring if not already running
            if not self._monitor_running:
                self.start_monitoring()

            _log.info(f"Order submitted: {order.id} {order.side} {order.quantity} {order.symbol}")

            # Notify callbacks
            self._notify_callbacks(order, "submitted")

            return True

        except (APIError, TimeoutError, ConnectionError) as e:
            _log.error(
                "ORDER_API_FAILED",
                extra={
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                    "op": "submit",
                    "symbol": order.symbol,
                    "qty": order.quantity,
                    "side": getattr(order.side, "value", order.side),
                    "type": getattr(order.order_type, "value", order.order_type),
                },
            )  # AI-AGENT-REF: structured submit failure
            order.status = OrderStatus.REJECTED
            order.notes += f" | Error: {e}"
            return False

    def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """Cancel an active order."""
        if not order_id:
            _log.warning("CANCEL_SKIPPED", extra={"reason": "empty_order_id"})
            return False
        try:
            order = self.active_orders.get(order_id)
            if not order:
                _log.warning(f"Cannot cancel order {order_id}: not found in active orders")
                return False

            success = order.cancel(reason)
            if success:
                self.active_orders.pop(order_id, None)
                self._notify_callbacks(order, "cancelled")

            return success

        except (APIError, TimeoutError, ConnectionError) as e:
            _log.error(
                "ORDER_API_FAILED",
                extra={
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                    "op": "cancel",
                    "order_id": order_id,
                },
            )  # AI-AGENT-REF: structured cancel failure
            return False

    def get_order_status(self, order_id: str) -> dict | None:
        """Get current order status."""
        order = self.orders.get(order_id)
        if order:
            return order.to_dict()
        return None

    def get_active_orders(self) -> list[dict]:
        """Get all active orders."""
        return [order.to_dict() for order in self.active_orders.values()]

    def get_order_history(self, symbol: str = None, limit: int = 100) -> list[dict]:
        """Get order history with optional filtering."""
        orders = list(self.orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        # Sort by creation time, most recent first
        orders.sort(key=lambda x: x.created_at, reverse=True)

        return [order.to_dict() for order in orders[:limit]]

    def add_execution_callback(self, callback: Callable):
        """Add callback for execution events."""
        self.execution_callbacks.append(callback)

    def start_monitoring(self):
        """Start order monitoring thread."""
        if self._monitor_running:
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_orders, daemon=True)
        self._monitor_thread.start()
        _log.info("Order monitoring started")

    def stop_monitoring(self):
        """Stop order monitoring thread."""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        _log.info("Order monitoring stopped")

    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission."""
        try:
            # Basic validation
            if not order.symbol or order.quantity <= 0:
                _log.error(
                    f"Invalid order parameters: symbol={order.symbol}, quantity={order.quantity}"
                )
                return False

            # AI-AGENT-REF: Quantize quantity to lot size
            tick = get_tick_size(order.symbol)
            lot = get_lot_size(order.symbol)

            # Ensure quantity is rounded to lot size
            original_quantity = order.quantity
            order.quantity = round_to_lot(order.quantity, lot)
            if original_quantity != order.quantity:
                _log.debug(
                    f"Quantity adjusted for {order.symbol}: {original_quantity} -> {order.quantity} (lot={lot})"
                )

            # Price validation and quantization for limit orders
            if order.order_type == OrderType.LIMIT:
                if not order.price or order.price <= 0:
                    _log.error(f"Limit order requires valid price: {order.price}")
                    return False

                # AI-AGENT-REF: Quantize price to tick size using Money precision
                if not isinstance(order.price, Money):
                    order.price = Money(order.price)

                original_price = order.price
                order.price = round_to_tick(order.price, tick)
                if float(original_price) != float(order.price):
                    _log.debug(
                        f"Price adjusted for {order.symbol}: {original_price} -> {order.price} (tick={tick})"
                    )

            # Side validation
            if order.side not in [OrderSide.BUY, OrderSide.SELL]:
                _log.error(f"Invalid order side: {order.side}")
                return False

            return True

        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            _log.error(
                "ORDER_VALIDATION_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )  # AI-AGENT-REF: narrow validation errors
            return False

    def _monitor_orders(self):
        """Monitor active orders for timeouts and updates."""
        while self._monitor_running:
            try:
                current_time = datetime.now(UTC)
                expired_orders = []

                for order_id, order in list(self.active_orders.items()):
                    # Check for timeout
                    age_seconds = (current_time - order.created_at).total_seconds()
                    if age_seconds > self.order_timeout:
                        expired_orders.append(order_id)

                    # Check if order is complete
                    if order.status in [
                        OrderStatus.FILLED,
                        OrderStatus.CANCELED,
                        OrderStatus.REJECTED,
                    ]:
                        self.active_orders.pop(order_id, None)
                        self._notify_callbacks(order, "completed")

                # Handle expired orders
                for order_id in expired_orders:
                    order = self.active_orders.get(order_id)
                    if order:
                        order.status = OrderStatus.EXPIRED
                        order.updated_at = current_time
                        self.active_orders.pop(order_id, None)
                        _log.warning(f"Order {order_id} expired after {self.order_timeout} seconds")
                        self._notify_callbacks(order, "expired")

                # AI-AGENT-REF: Run reconciliation after order processing
                from .reconcile import reconcile_positions_and_orders

                reconcile_positions_and_orders()

                time.sleep(1)  # Check every second

            except (APIError, TimeoutError, ConnectionError) as e:
                _log.error(
                    "ORDER_MONITOR_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
                time.sleep(5)  # Back off on error

    def _notify_callbacks(self, order: Order, event_type: str):
        """Notify registered callbacks of order events."""
        try:
            for callback in self.execution_callbacks:
                try:
                    callback(order, event_type)
                except (
                    KeyError,
                    ValueError,
                    TypeError,
                    RuntimeError,
                ) as e:
                    _log.error(
                        "CALLBACK_FAILED",
                        extra={
                            "cause": e.__class__.__name__,
                            "detail": str(e),
                            "order_id": order.id,
                        },
                    )
        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            _log.error(
                "CALLBACK_NOTIFICATION_FAILED",
                extra={
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                    "order_id": order.id,
                },
            )  # AI-AGENT-REF: narrow callback errors


class ExecutionEngine:
    """
    Main execution engine for institutional order processing.

    Coordinates order management, execution algorithms,
    and provides unified execution interface.
    """

    _minute_stats: dict[str, float] = {}  # AI-AGENT-REF: patchable minute stats
    _latest_quote: dict[str, float] = {}  # AI-AGENT-REF: patchable quote cache

    def __init__(self, market_data_feed=None, broker_interface=None):
        """Initialize execution engine."""
        # AI-AGENT-REF: Main institutional execution engine
        self.order_manager = OrderManager()
        self.market_data_feed = market_data_feed
        self.broker_interface = broker_interface

        # Execution statistics
        self.execution_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0.0,
            "average_fill_time": 0.0,
        }

        _log.info("ExecutionEngine initialized")

    def _assess_liquidity(self, symbol: str, quantity: int) -> tuple[int, bool]:
        """Assess liquidity and optionally adjust quantity."""
        bid, ask = (0.0, 0.0)
        try:
            bid, ask = self._latest_quote()
        except Exception:
            return quantity, False
        spread_pct = (ask - bid) / bid if bid else 0.0
        if spread_pct >= 0.01:
            return int(quantity * 0.75), False
        return quantity, False

    def _reconcile_partial_fills(
        self,
        *,
        symbol: str,
        submitted_qty: int,
        remaining_qty: int,
        side: str,
        last_order,
    ) -> None:
        """Stub for partial fill reconciliation."""
        return None

    def execute_order(
        self,
        symbol: str,
        quantity_or_side=None,
        side_or_quantity=None,
        order_type: OrderType = OrderType.MARKET,
        method=None,
        **kwargs,
    ):
        """
        Execute a trading order.

        Supports both new signature (symbol, side, quantity) and legacy signature (symbol, quantity, side).

        Args:
            symbol: Trading symbol
            quantity_or_side: Either quantity (legacy) or OrderSide (new)
            side_or_quantity: Either side string (legacy) or quantity int (new)
            order_type: Type of order
            method: Execution method (for legacy compatibility)
            **kwargs: Additional order parameters

        Returns:
            Order ID if successful, None if failed
        """
        try:
            # Handle legacy signature: execute_order(symbol, qty, side, method=...)
            if isinstance(quantity_or_side, int):
                # Legacy signature: symbol, qty, side
                quantity = quantity_or_side
                side_str = side_or_quantity
                if isinstance(side_str, str):
                    if side_str.lower() == "buy":
                        side = OrderSide.BUY
                    elif side_str.lower() == "sell":
                        side = OrderSide.SELL
                    else:
                        raise ValueError(f"Invalid side: {side_str}")
                else:
                    side = side_str
            else:
                # New signature: symbol, side, quantity
                side = quantity_or_side
                quantity = side_or_quantity

            # AI-AGENT-REF: call sliced execution directly for TWAP
            if method == "twap":
                return self.execute_sliced(symbol, quantity, side, **kwargs)

            quantity = int(_ensure_positive_qty(quantity))
            kwargs["limit_price"] = _ensure_valid_price(kwargs.get("limit_price"))
            kwargs["stop_price"] = _ensure_valid_price(kwargs.get("stop_price"))

            payload: dict[str, Any] = {
                "symbol": symbol,
                "side": getattr(side, "value", side),
                "qty": quantity,
                "type": getattr(order_type, "value", order_type),
                "time_in_force": kwargs.get("time_in_force"),
                "limit_price": kwargs.get("limit_price"),
                "stop_price": kwargs.get("stop_price"),
            }
            _log.debug(
                "ORDER_SUBMIT_PAYLOAD",
                extra={
                    k: payload.get(k)
                    for k in (
                        "symbol",
                        "side",
                        "qty",
                        "type",
                        "time_in_force",
                        "limit_price",
                        "stop_price",
                    )
                },
            )

            # Create order
            order = Order(symbol, side, quantity, order_type, **kwargs)

            # Submit order
            if self.order_manager.submit_order(order):
                self.execution_stats["total_orders"] += 1

                # Simulate execution for demo purposes
                if order_type == OrderType.MARKET:
                    self._simulate_market_execution(order)

                return order.id
            else:
                self.execution_stats["rejected_orders"] += 1
                return None

        except (ValueError, TypeError, KeyError) as e:
            _log.error(
                "EXECUTE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )  # AI-AGENT-REF: narrow execution errors
            self.execution_stats["rejected_orders"] += 1
            raise

    def execute_sliced(self, symbol: str, quantity: int, side: OrderSide, **kwargs):
        """Execute order slices by delegating to execute_order."""
        # AI-AGENT-REF: delegate sliced execution to unified path
        return self.execute_order(symbol, side, quantity, **kwargs)

    def _simulate_market_execution(self, order: Order):
        """Simulate market order execution (demo purposes)."""
        try:
            # Simulate market price
            base_price = order.price or 100.0  # Default price if not provided

            # Simulate partial fills for large orders
            remaining = order.quantity
            while remaining > 0 and order.status != OrderStatus.CANCELED:
                fill_quantity = min(remaining, max(1, remaining // 3))  # Fill in chunks
                fill_price = base_price * (
                    1 + (hash(order.id) % 100 - 50) / 10000
                )  # Small random variation

                order.add_fill(fill_quantity, fill_price)
                remaining -= fill_quantity

                if remaining > 0:
                    time.sleep(0.1)  # Simulate execution delay

            if order.is_filled:
                self.execution_stats["filled_orders"] += 1
                self.execution_stats["total_volume"] += order.notional_value

                fill_time = (order.executed_at - order.created_at).total_seconds()
                self.execution_stats["average_fill_time"] = (
                    self.execution_stats["average_fill_time"]
                    * (self.execution_stats["filled_orders"] - 1)
                    + fill_time
                ) / self.execution_stats["filled_orders"]

        except (
            KeyError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            _log.error(
                "SIMULATION_FAILED",
                extra={
                    "cause": e.__class__.__name__,
                    "detail": str(e),
                    "order_id": order.id,
                },
            )  # AI-AGENT-REF: narrow simulation errors

    def get_execution_stats(self) -> dict:
        """Get execution engine statistics."""
        stats = self.execution_stats.copy()
        stats["active_orders"] = len(self.order_manager.active_orders)
        stats["success_rate"] = (
            stats["filled_orders"] / stats["total_orders"] if stats["total_orders"] > 0 else 0
        )
        return stats
