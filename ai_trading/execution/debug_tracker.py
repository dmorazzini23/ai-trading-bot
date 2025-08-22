"""Enhanced trade execution debugging and tracking system.

This module provides comprehensive logging and tracking for the complete
signal-to-execution pipeline, including correlation IDs, order lifecycle
tracking, and detailed execution logging.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any

from ai_trading.logging import get_logger


def get_phase_logger(name: str, phase: str) -> logging.Logger:
    """Get a logger for a specific phase - fallback implementation."""
    logger_name = f"{name}.{phase}" if phase else name
    return get_logger(logger_name)


class ExecutionPhase(Enum):
    """Phases of order execution lifecycle."""

    SIGNAL_GENERATED = "signal_generated"
    RISK_CHECK = "risk_check"
    ORDER_PREPARED = "order_prepared"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACKNOWLEDGED = "order_acknowledged"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_UPDATED = "position_updated"
    PNL_CALCULATED = "pnl_calculated"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ExecutionDebugTracker:
    """Comprehensive execution debugging and correlation tracking."""

    def __init__(self):
        self.logger = get_phase_logger(__name__, "EXEC_DEBUG")
        self._lock = Lock()

        # Track active orders by correlation ID
        self._active_orders: dict[str, dict[str, Any]] = {}

        # Track execution timeline for each correlation ID
        self._execution_timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Track order lifecycle events
        self._order_events: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Track recent execution statistics
        self._recent_executions: deque = deque(maxlen=1000)

        # Track failed executions for analysis
        self._failed_executions: deque = deque(maxlen=500)

        # Track position updates
        self._position_updates: deque = deque(maxlen=500)

        # Debug flags
        self.verbose_logging = False
        self.trace_mode = False

    def generate_correlation_id(self, symbol: str, side: str) -> str:
        """Generate unique correlation ID for tracking order lifecycle."""
        timestamp = int(time.time() * 1000)  # milliseconds
        unique_id = str(uuid.uuid4())[:8]
        return f"{symbol}_{side}_{timestamp}_{unique_id}"

    def start_execution_tracking(
        self,
        correlation_id: str,
        symbol: str,
        qty: int,
        side: str,
        signal_data: dict | None = None,
    ) -> None:
        """Start tracking a new order execution."""
        execution_start = {
            "correlation_id": correlation_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "start_time": datetime.now(UTC).isoformat(),
            "signal_data": signal_data or {},
            "status": OrderStatus.PENDING.value,
            "phases": [],
        }

        # AI-AGENT-REF: Use timeout-based lock to prevent deadlock
        lock_acquired = False
        try:
            lock_acquired = self._lock.acquire(timeout=5.0)
            if lock_acquired:
                self._active_orders[correlation_id] = execution_start
        except (ValueError, TypeError) as e:
            self.logger.error(
                "START_TRACKING_ERROR",
                extra={"correlation_id": correlation_id, "error": str(e)},
            )
        finally:
            if lock_acquired:
                self._lock.release()

        # AI-AGENT-REF: Call log_execution_event outside of lock to prevent recursive deadlock
        self.log_execution_event(
            correlation_id,
            ExecutionPhase.SIGNAL_GENERATED,
            {"symbol": symbol, "qty": qty, "side": side, "signal_data": signal_data},
        )

    def log_execution_event(
        self,
        correlation_id: str,
        phase: ExecutionPhase,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Log an execution phase event with correlation ID."""
        timestamp = datetime.now(UTC).isoformat()

        event = {
            "timestamp": timestamp,
            "correlation_id": correlation_id,
            "phase": phase.value,
            "data": data or {},
        }

        # AI-AGENT-REF: Use timeout-based lock acquisition to prevent deadlock
        lock_acquired = False
        try:
            lock_acquired = self._lock.acquire(timeout=5.0)  # 5 second timeout
            if not lock_acquired:
                # Fallback: log without state update to prevent blocking
                self.logger.warning(
                    "LOCK_TIMEOUT_EXECUTION_EVENT",
                    extra={
                        "correlation_id": correlation_id,
                        "phase": phase.value,
                        "message": "Failed to acquire lock within timeout, logging without state update",
                    },
                )
                return

            self._execution_timelines[correlation_id].append(event)

            # Update active order status if relevant
            if correlation_id in self._active_orders:
                self._active_orders[correlation_id]["phases"].append(event)

                # Update status based on phase
                if phase == ExecutionPhase.ORDER_SUBMITTED:
                    self._active_orders[correlation_id][
                        "status"
                    ] = OrderStatus.SUBMITTED.value
                elif phase == ExecutionPhase.ORDER_ACKNOWLEDGED:
                    self._active_orders[correlation_id][
                        "status"
                    ] = OrderStatus.ACKNOWLEDGED.value
                elif phase == ExecutionPhase.ORDER_FILLED:
                    self._active_orders[correlation_id][
                        "status"
                    ] = OrderStatus.FILLED.value
                elif phase == ExecutionPhase.ORDER_PARTIALLY_FILLED:
                    self._active_orders[correlation_id][
                        "status"
                    ] = OrderStatus.PARTIALLY_FILLED.value
                elif phase == ExecutionPhase.ORDER_REJECTED:
                    self._active_orders[correlation_id][
                        "status"
                    ] = OrderStatus.REJECTED.value
                elif phase == ExecutionPhase.ORDER_CANCELLED:
                    self._active_orders[correlation_id][
                        "status"
                    ] = OrderStatus.CANCELLED.value
        except (ValueError, TypeError) as e:
            # AI-AGENT-REF: Graceful error handling for lock operations
            self.logger.error(
                "EXECUTION_EVENT_ERROR",
                extra={
                    "correlation_id": correlation_id,
                    "phase": phase.value,
                    "error": str(e),
                    "message": "Error updating execution event state",
                },
            )
        finally:
            if lock_acquired:
                self._lock.release()

        # Log the event (moved outside lock to prevent circular logging)
        log_data = {
            "correlation_id": correlation_id,
            "phase": phase.value,
            "timestamp": timestamp,
        }
        log_data.update(data or {})

        try:
            if self.verbose_logging or self.trace_mode:
                self.logger.info(f"EXEC_EVENT_{phase.value.upper()}", extra=log_data)
            elif phase in [
                ExecutionPhase.SIGNAL_GENERATED,
                ExecutionPhase.ORDER_SUBMITTED,
                ExecutionPhase.ORDER_FILLED,
                ExecutionPhase.ORDER_REJECTED,
            ]:
                # Log only key phases in normal mode (but not if already logged in verbose mode)
                self.logger.info(f"EXEC_EVENT_{phase.value.upper()}", extra=log_data)
        except (ValueError, TypeError, AttributeError) as e:
            # AI-AGENT-REF: Prevent logging errors from cascading
            self.logger.exception(
                "debug_tracker: logging error during execution event", exc_info=e
            )

    def log_order_result(
        self,
        correlation_id: str,
        success: bool,
        order_data: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Log the final result of an order execution."""
        timestamp = datetime.now(UTC).isoformat()

        result_data = {
            "correlation_id": correlation_id,
            "success": success,
            "timestamp": timestamp,
            "order_data": order_data or {},
            "error": error,
        }

        # AI-AGENT-REF: Use timeout-based lock to prevent deadlock
        lock_acquired = False
        order_info = None
        found_order = False

        try:
            lock_acquired = self._lock.acquire(timeout=5.0)
            if lock_acquired and correlation_id in self._active_orders:
                order_info = self._active_orders[correlation_id].copy()
                order_info.update(result_data)
                found_order = True

                if success:
                    self._recent_executions.append(order_info)
                else:
                    self._failed_executions.append(order_info)

                # Remove from active orders
                del self._active_orders[correlation_id]
        except (ValueError, TypeError) as e:
            self.logger.error(
                "ORDER_RESULT_ERROR",
                extra={"correlation_id": correlation_id, "error": str(e)},
            )
        finally:
            if lock_acquired:
                self._lock.release()

        # AI-AGENT-REF: Log outside of lock to prevent circular logging deadlock
        try:
            if found_order:
                if success:
                    self.logger.info("ORDER_EXECUTION_SUCCESS", extra=result_data)
                else:
                    self.logger.error("ORDER_EXECUTION_FAILED", extra=result_data)
            else:
                self.logger.warning(
                    "UNKNOWN_CORRELATION_ID",
                    extra={
                        "correlation_id": correlation_id,
                        "message": "Attempted to log result for unknown correlation ID",
                    },
                )
        except (ValueError, TypeError, AttributeError) as e:
            # AI-AGENT-REF: Prevent logging errors from cascading
            self.logger.exception(
                "debug_tracker: logging error during order result", exc_info=e
            )

    def log_position_update(
        self,
        symbol: str,
        old_qty: float,
        new_qty: float,
        correlation_id: str | None = None,
    ) -> None:
        """Log position updates with optional correlation to order."""
        position_update = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "old_qty": old_qty,
            "new_qty": new_qty,
            "qty_change": new_qty - old_qty,
            "correlation_id": correlation_id,
        }

        with self._lock:
            self._position_updates.append(position_update)

        if correlation_id:
            self.log_execution_event(
                correlation_id, ExecutionPhase.POSITION_UPDATED, position_update
            )

        self.logger.info("POSITION_UPDATE", extra=position_update)

    def get_active_orders(self) -> dict[str, dict[str, Any]]:
        """Get all currently active orders being tracked."""
        with self._lock:
            return self._active_orders.copy()

    def get_execution_timeline(self, correlation_id: str) -> list[dict[str, Any]]:
        """Get the complete execution timeline for a correlation ID."""
        with self._lock:
            return self._execution_timelines[correlation_id].copy()

    def get_recent_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent successful executions."""
        with self._lock:
            return list(self._recent_executions)[-limit:]

    def get_failed_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent failed executions for analysis."""
        with self._lock:
            return list(self._failed_executions)[-limit:]

    def get_position_updates(
        self, symbol: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get recent position updates, optionally filtered by symbol."""
        with self._lock:
            updates = list(self._position_updates)[-limit:]
            if symbol:
                updates = [u for u in updates if u["symbol"] == symbol]
            return updates

    def set_debug_mode(self, verbose: bool = True, trace: bool = False) -> None:
        """Enable/disable debug logging modes."""
        self.verbose_logging = verbose
        self.trace_mode = trace

        mode = "TRACE" if trace else "VERBOSE" if verbose else "NORMAL"
        self.logger.info("DEBUG_MODE_CHANGED", extra={"mode": mode})

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics for monitoring."""
        with self._lock:
            active_count = len(self._active_orders)
            recent_success_count = len(self._recent_executions)
            recent_failure_count = len(self._failed_executions)

            # Calculate success rate from recent executions
            total_recent = recent_success_count + recent_failure_count
            success_rate = (
                recent_success_count / total_recent if total_recent > 0 else 0
            )

            # Get status breakdown of active orders
            status_breakdown = {}
            for order in self._active_orders.values():
                status = order.get("status", "unknown")
                status_breakdown[status] = status_breakdown.get(status, 0) + 1

            return {
                "active_orders": active_count,
                "recent_successes": recent_success_count,
                "recent_failures": recent_failure_count,
                "success_rate": success_rate,
                "status_breakdown": status_breakdown,
                "position_updates_count": len(self._position_updates),
            }


# Global debug tracker instance
_debug_tracker: ExecutionDebugTracker | None = None
_tracker_lock = Lock()


def get_debug_tracker() -> ExecutionDebugTracker:
    """Get or create the global debug tracker instance."""
    global _debug_tracker
    with _tracker_lock:
        if _debug_tracker is None:
            _debug_tracker = ExecutionDebugTracker()
        return _debug_tracker


def enable_debug_mode(verbose: bool = True, trace: bool = False) -> None:
    """Enable debug mode for execution tracking."""
    tracker = get_debug_tracker()
    tracker.set_debug_mode(verbose, trace)


def log_signal_to_execution(
    symbol: str, side: str, qty: int, signal_data: dict | None = None
) -> str:
    """Start tracking a signal-to-execution flow and return correlation ID."""
    tracker = get_debug_tracker()
    correlation_id = tracker.generate_correlation_id(symbol, side)
    tracker.start_execution_tracking(correlation_id, symbol, qty, side, signal_data)
    return correlation_id


def log_execution_phase(
    correlation_id: str, phase: ExecutionPhase, data: dict | None = None
) -> None:
    """Log an execution phase with correlation ID."""
    tracker = get_debug_tracker()
    tracker.log_execution_event(correlation_id, phase, data)


def log_order_outcome(
    correlation_id: str,
    success: bool,
    order_data: dict | None = None,
    error: str | None = None,
) -> None:
    """Log the final outcome of an order execution."""
    tracker = get_debug_tracker()
    tracker.log_order_result(correlation_id, success, order_data, error)


def log_position_change(
    symbol: str, old_qty: float, new_qty: float, correlation_id: str | None = None
) -> None:
    """Log a position change with optional correlation to order."""
    tracker = get_debug_tracker()
    tracker.log_position_update(symbol, old_qty, new_qty, correlation_id)


def get_execution_statistics() -> dict[str, Any]:
    """Get current execution statistics."""
    tracker = get_debug_tracker()
    return tracker.get_execution_stats()
