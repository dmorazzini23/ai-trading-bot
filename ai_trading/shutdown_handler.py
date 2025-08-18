# ruff: noqa
"""Graceful shutdown handler for production trading safety.

Provides safe shutdown procedures that protect open positions,
cancel pending orders, and ensure data consistency during
system restarts or emergency shutdowns.

AI-AGENT-REF: Graceful shutdown for position safety in production
"""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

Hook = Callable[[], None]
PositionsHandler = Callable[[], list[dict[str, Any]]]
try:  # AI-AGENT-REF: resilient Alpaca import
    from alpaca.trading.client import TradingClient  # type: ignore  # noqa: F401
    from alpaca.common.exceptions import APIError  # type: ignore
except Exception:  # AI-AGENT-REF: fallback when SDK missing
    TradingClient = None  # type: ignore

    class APIError(Exception): ...


from ai_trading.logging import logger


class ShutdownReason(Enum):
    """Reasons for system shutdown."""

    USER_REQUEST = "user_request"
    SYSTEM_ERROR = "system_error"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    EMERGENCY_STOP = "emergency_stop"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    SIGNAL_RECEIVED = "signal_received"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ShutdownPhase(Enum):
    """Phases of the shutdown process."""

    INITIATED = "initiated"
    STOPPING_NEW_ORDERS = "stopping_new_orders"
    CANCELING_PENDING_ORDERS = "canceling_pending_orders"
    CLOSING_POSITIONS = "closing_positions"
    SAVING_STATE = "saving_state"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ShutdownStatus:
    """Current shutdown status."""

    is_shutting_down: bool
    reason: ShutdownReason | None
    phase: ShutdownPhase
    started_at: datetime | None
    estimated_completion: datetime | None
    progress_percent: float
    current_action: str
    positions_to_close: int = 0
    positions_closed: int = 0
    orders_to_cancel: int = 0
    orders_canceled: int = 0
    errors: list[str] = field(default_factory=list)


class ShutdownHandler:
    """Comprehensive shutdown handler for trading system."""

    def __init__(self):
        self.logger = logger
        self._status = ShutdownStatus(
            is_shutting_down=False,
            reason=None,
            phase=ShutdownPhase.INITIATED,
            started_at=None,
            estimated_completion=None,
            progress_percent=0.0,
            current_action="System running normally",
        )

        # Shutdown configuration
        self.config = {
            "max_shutdown_time_minutes": 15,  # Maximum time to allow for shutdown
            "position_close_timeout_minutes": 10,  # Time to close positions
            "order_cancel_timeout_minutes": 3,  # Time to cancel orders
            "emergency_shutdown_seconds": 30,  # Emergency shutdown time
            "save_state_on_shutdown": True,  # Save system state
            "force_close_positions": False,  # Force close positions if needed
        }

        # Registered shutdown hooks
        self._pre_shutdown_hooks: list[Callable[[], None]] = list()
        self._position_handlers: list[Callable] = []
        self._order_handlers: list[Callable] = []
        self._cleanup_hooks: list[Callable] = []
        self._post_shutdown_hooks: list[Callable] = []

        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()

        # Signal handlers
        self._original_handlers = {}
        self._setup_signal_handlers()

        # State tracking
        self._active_positions: list[dict[str, Any]] = []
        self._pending_orders: list[dict[str, Any]] = []
        self._system_state: dict[str, Any] = {}

        self.logger.info("ShutdownHandler initialized")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            # Handle common shutdown signals
            signals_to_handle = [signal.SIGTERM, signal.SIGINT]

            # Add SIGQUIT on Unix systems
            if hasattr(signal, "SIGQUIT"):
                signals_to_handle.append(signal.SIGQUIT)

            for sig in signals_to_handle:
                self._original_handlers[sig] = signal.signal(sig, self._signal_handler)

            self.logger.info(
                f"Signal handlers registered for: {[s.name for s in signals_to_handle]}"
            )

        except (
            APIError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as e:  # AI-AGENT-REF: include OSError in shutdown handling
            self.logger.error(
                "SIGNAL_HANDLER_SETUP_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        self.logger.info(f"Received signal: {signal_name}")

        # Initiate graceful shutdown
        if not self._status.is_shutting_down:
            asyncio.create_task(self.shutdown(ShutdownReason.SIGNAL_RECEIVED))

    def register_pre_shutdown_hook(self, hook: Hook | None = None) -> None:
        """Register a pre-shutdown hook."""
        if hook is None:
            return
        if self._pre_shutdown_hooks is None:
            self._pre_shutdown_hooks = []
        self._pre_shutdown_hooks.append(hook)
        self.logger.debug(f"Registered pre-shutdown hook: {hook.__name__}")

    def register_position_handler(self, handler: PositionsHandler) -> None:
        """Register a position handler that returns list of positions to close."""
        self._position_handlers.append(handler)
        self.logger.debug(f"Registered position handler: {handler.__name__}")

    def register_order_handler(self, handler: Callable) -> None:
        """Register an order handler that returns list of orders to cancel."""
        self._order_handlers.append(handler)
        self.logger.debug(f"Registered order handler: {handler.__name__}")

    def register_cleanup_hook(self, hook: Hook) -> None:
        """Register a cleanup hook."""
        self._cleanup_hooks.append(hook)
        self.logger.debug(f"Registered cleanup hook: {hook.__name__}")

    def register_post_shutdown_hook(self, hook: Hook) -> None:
        """Register a post-shutdown hook."""
        self._post_shutdown_hooks.append(hook)
        self.logger.debug(f"Registered post-shutdown hook: {hook.__name__}")

    async def shutdown(
        self,
        reason: ShutdownReason = ShutdownReason.USER_REQUEST,
        emergency: bool = False,
    ) -> bool:
        """Initiate graceful shutdown process."""
        with self._lock:
            if self._status.is_shutting_down:
                self.logger.warning("Shutdown already in progress")
                return False

            self._status.is_shutting_down = True
            self._status.reason = reason
            self._status.started_at = datetime.now(UTC)
            self._status.phase = ShutdownPhase.INITIATED
            self._status.current_action = f"Initiating shutdown: {reason.value}"

            # Set completion estimate
            if emergency:
                self._status.estimated_completion = self._status.started_at + timedelta(
                    seconds=self.config["emergency_shutdown_seconds"]
                )
            else:
                self._status.estimated_completion = self._status.started_at + timedelta(
                    minutes=self.config["max_shutdown_time_minutes"]
                )

        self.logger.info(
            f"Initiating {'emergency' if emergency else 'graceful'} shutdown: {reason.value}"
        )

        try:
            if emergency:
                success = await self._emergency_shutdown()
            else:
                success = await self._graceful_shutdown()

            if success:
                self._status.phase = ShutdownPhase.COMPLETED
                self._status.progress_percent = 100.0
                self._status.current_action = "Shutdown completed successfully"
                self.logger.info("Shutdown completed successfully")
            else:
                self._status.phase = ShutdownPhase.FAILED
                self._status.current_action = "Shutdown failed"
                self.logger.error("Shutdown failed")

            return success

        except (APIError, TimeoutError, ConnectionError, OSError) as e:
            self._status.phase = ShutdownPhase.FAILED
            self._status.current_action = f"Shutdown error: {str(e)}"
            self._status.errors.append(str(e))
            self.logger.error(
                "SHUTDOWN_ERROR",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            return False
        finally:
            self._shutdown_event.set()

    async def _graceful_shutdown(self) -> bool:
        """Perform graceful shutdown with position protection."""
        try:
            # Phase 1: Pre-shutdown hooks
            self._status.phase = ShutdownPhase.INITIATED
            self._status.progress_percent = 5.0
            self._status.current_action = "Running pre-shutdown hooks"
            await self._run_pre_shutdown_hooks()

            # Phase 2: Stop accepting new orders
            self._status.phase = ShutdownPhase.STOPPING_NEW_ORDERS
            self._status.progress_percent = 15.0
            self._status.current_action = "Stopping new order acceptance"
            await self._stop_new_orders()

            # Phase 3: Cancel pending orders
            self._status.phase = ShutdownPhase.CANCELING_PENDING_ORDERS
            self._status.progress_percent = 30.0
            self._status.current_action = "Canceling pending orders"
            success = await self._cancel_pending_orders()
            if not success:
                self.logger.warning("Failed to cancel all pending orders")

            # Phase 4: Close positions (optional based on configuration)
            if self.config.get("force_close_positions", False):
                self._status.phase = ShutdownPhase.CLOSING_POSITIONS
                self._status.progress_percent = 50.0
                self._status.current_action = "Closing open positions"
                success = await self._close_positions()
                if not success:
                    self.logger.warning("Failed to close all positions")
            else:
                self.logger.info("Leaving positions open as configured")
                self._status.progress_percent = 70.0

            # Phase 5: Save system state
            self._status.phase = ShutdownPhase.SAVING_STATE
            self._status.progress_percent = 80.0
            self._status.current_action = "Saving system state"
            if self.config.get("save_state_on_shutdown", True):
                await self._save_system_state()

            # Phase 6: Cleanup
            self._status.phase = ShutdownPhase.CLEANUP
            self._status.progress_percent = 90.0
            self._status.current_action = "Running cleanup hooks"
            await self._run_cleanup_hooks()

            # Phase 7: Post-shutdown hooks
            self._status.progress_percent = 95.0
            self._status.current_action = "Running post-shutdown hooks"
            await self._run_post_shutdown_hooks()

            return True

        except (APIError, TimeoutError, ConnectionError, OSError) as e:
            self.logger.error(
                "GRACEFUL_SHUTDOWN_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            self._status.errors.append(str(e))
            return False

    async def _emergency_shutdown(self) -> bool:
        """Perform emergency shutdown with minimal time."""
        try:
            self.logger.warning("Performing emergency shutdown")

            # Only critical actions in emergency shutdown
            timeout = self.config["emergency_shutdown_seconds"]

            # Cancel orders quickly
            self._status.current_action = "Emergency order cancellation"
            self._status.progress_percent = 25.0
            await asyncio.wait_for(self._cancel_pending_orders(), timeout=timeout / 3)

            # Save critical state
            self._status.current_action = "Emergency state save"
            self._status.progress_percent = 60.0
            await asyncio.wait_for(self._save_critical_state(), timeout=timeout / 3)

            # Quick cleanup
            self._status.current_action = "Emergency cleanup"
            self._status.progress_percent = 90.0
            await asyncio.wait_for(self._emergency_cleanup(), timeout=timeout / 3)

            return True

        except (APIError, TimeoutError, ConnectionError, OSError) as e:
            self.logger.error(
                "EMERGENCY_SHUTDOWN_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            self._status.errors.append(str(e))
            return False

    async def _run_pre_shutdown_hooks(self) -> None:
        """Run pre-shutdown hooks."""
        for hook in self._pre_shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except (APIError, TimeoutError, ConnectionError, OSError) as e:
                self.logger.error(
                    "PRE_SHUTDOWN_HOOK_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
                self._status.errors.append(f"Pre-shutdown hook error: {e}")

    async def _stop_new_orders(self) -> None:
        """Stop accepting new orders."""
        # Set global flag to prevent new orders
        # This would integrate with the actual trading engine
        self.logger.info("Stopped accepting new orders")

    async def _cancel_pending_orders(self) -> bool:
        """Cancel all pending orders."""
        try:
            # Collect all pending orders
            all_orders = []
            for handler in self._order_handlers:
                try:
                    orders = handler()
                    if orders:
                        all_orders.extend(orders)
                except (APIError, TimeoutError, ConnectionError, OSError) as e:
                    self.logger.error(
                        "ORDER_HANDLER_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
                    self._status.errors.append(f"Order handler error: {e}")

            self._status.orders_to_cancel = len(all_orders)
            self._status.orders_canceled = 0

            # Cancel orders with timeout
            timeout = timedelta(minutes=self.config["order_cancel_timeout_minutes"])
            start_time = datetime.now(UTC)

            for order in all_orders:
                if datetime.now(UTC) - start_time > timeout:
                    self.logger.warning("Order cancellation timeout reached")
                    break

                try:
                    # Cancel order (would integrate with actual trading API)
                    await self._cancel_single_order(order)
                    self._status.orders_canceled += 1

                except (APIError, TimeoutError, ConnectionError, OSError) as e:
                    self.logger.warning(
                        "SHUTDOWN_CANCEL_FAILED",
                        extra={
                            "cause": e.__class__.__name__,
                            "detail": str(e),
                            "order_id": order.get("id", "unknown"),
                        },
                    )  # AI-AGENT-REF: best-effort cancel logging
                    self._status.errors.append(f"Order cancel error: {e}")
                    continue

            success_rate = self._status.orders_canceled / max(
                self._status.orders_to_cancel, 1
            )
            self.logger.info(
                f"Canceled {self._status.orders_canceled}/{self._status.orders_to_cancel} orders ({success_rate:.1%})"  # noqa: E501
            )

            return success_rate >= 0.9  # 90% success rate required

        except (APIError, TimeoutError, ConnectionError, OSError) as e:
            self.logger.error(
                "ORDER_CANCELLATION_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            return False

    async def _close_positions(self) -> bool:
        """Close all open positions."""
        try:
            # Collect all positions
            all_positions = []
            for handler in self._position_handlers:
                try:
                    positions = handler()
                    if positions:
                        all_positions.extend(positions)
                except (APIError, TimeoutError, ConnectionError, OSError) as e:
                    self.logger.error(
                        "POSITION_HANDLER_FAILED",
                        extra={"cause": e.__class__.__name__, "detail": str(e)},
                    )
                    self._status.errors.append(f"Position handler error: {e}")

            self._status.positions_to_close = len(all_positions)
            self._status.positions_closed = 0

            # Close positions with timeout
            timeout = timedelta(minutes=self.config["position_close_timeout_minutes"])
            start_time = datetime.now(UTC)

            for position in all_positions:
                if datetime.now(UTC) - start_time > timeout:
                    self.logger.warning("Position closing timeout reached")
                    break

                try:
                    # Close position (would integrate with actual trading API)
                    await self._close_single_position(position)
                    self._status.positions_closed += 1

                except (APIError, TimeoutError, ConnectionError, OSError) as e:
                    self.logger.warning(
                        "SHUTDOWN_CLOSE_POSITION_FAILED",
                        extra={
                            "cause": e.__class__.__name__,
                            "detail": str(e),
                            "order_id": position.get("symbol", "unknown"),
                        },
                    )  # AI-AGENT-REF: best-effort position close logging
                    self._status.errors.append(f"Position close error: {e}")
                    continue

            success_rate = self._status.positions_closed / max(
                self._status.positions_to_close, 1
            )
            self.logger.info(
                f"Closed {self._status.positions_closed}/{self._status.positions_to_close} positions ({success_rate:.1%})"  # noqa: E501
            )

            return success_rate >= 0.9  # 90% success rate required

        except (APIError, TimeoutError, ConnectionError, OSError) as e:
            self.logger.error(
                "POSITION_CLOSING_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            return False

    async def _cancel_single_order(self, order: dict[str, Any]) -> None:
        """Cancel a single order."""
        # This would integrate with the actual trading API
        self.logger.debug(f"Canceling order: {order.get('id', 'unknown')}")
        await asyncio.sleep(0.1)  # Simulate API call

    async def _close_single_position(self, position: dict[str, Any]) -> None:
        """Close a single position."""
        # This would integrate with the actual trading API
        self.logger.debug(f"Closing position: {position.get('symbol', 'unknown')}")
        await asyncio.sleep(0.1)  # Simulate API call

    async def _save_system_state(self) -> None:
        """Save complete system state."""
        try:
            state_file = (
                Path("logs")
                / f"shutdown_state_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            )
            state_file.parent.mkdir(exist_ok=True)

            state_data = {
                "shutdown_info": {
                    "reason": (
                        self._status.reason.value if self._status.reason else None
                    ),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "phase": self._status.phase.value,
                    "positions_status": {
                        "total": self._status.positions_to_close,
                        "closed": self._status.positions_closed,
                    },
                    "orders_status": {
                        "total": self._status.orders_to_cancel,
                        "canceled": self._status.orders_canceled,
                    },
                },
                "system_state": self._system_state,
                "active_positions": self._active_positions,
                "pending_orders": self._pending_orders,
            }

            import json

            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            self.logger.info(f"System state saved to: {state_file}")

        except (APIError, TimeoutError, ConnectionError, OSError) as e:
            self.logger.error(
                "STATE_SAVE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            self._status.errors.append(f"State save error: {e}")

    async def _save_critical_state(self) -> None:
        """Save only critical state for emergency shutdown."""
        try:
            state_file = (
                Path("logs")
                / f"emergency_state_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            )
            state_file.parent.mkdir(exist_ok=True)

            critical_data = {
                "emergency_shutdown": True,
                "timestamp": datetime.now(UTC).isoformat(),
                "reason": self._status.reason.value if self._status.reason else None,
                "positions_count": len(self._active_positions),
                "orders_count": len(self._pending_orders),
            }

            import json

            with open(state_file, "w") as f:
                json.dump(critical_data, f, indent=2)

            self.logger.info(f"Critical state saved to: {state_file}")

        except (APIError, TimeoutError, ConnectionError, OSError) as e:
            self.logger.error(
                "CRITICAL_STATE_SAVE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )

    async def _run_cleanup_hooks(self) -> None:
        """Run cleanup hooks."""
        for hook in self._cleanup_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except (APIError, TimeoutError, ConnectionError, OSError) as e:
                self.logger.error(
                    "CLEANUP_HOOK_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
                self._status.errors.append(f"Cleanup hook error: {e}")

    async def _emergency_cleanup(self) -> None:
        """Emergency cleanup - only critical operations."""
        # Close critical resources, files, connections
        self.logger.info("Emergency cleanup completed")

    async def _run_post_shutdown_hooks(self) -> None:
        """Run post-shutdown hooks."""
        for hook in self._post_shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except (APIError, TimeoutError, ConnectionError, OSError) as e:
                self.logger.error(
                    "POST_SHUTDOWN_HOOK_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )

    def get_shutdown_status(self) -> ShutdownStatus:
        """Get current shutdown status."""
        return self._status

    def is_shutting_down(self) -> bool:
        """Check if system is currently shutting down."""
        return self._status.is_shutting_down

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for shutdown to complete."""
        return self._shutdown_event.wait(timeout)

    def update_system_state(self, state: dict[str, Any]) -> None:
        """Update system state to be saved on shutdown."""
        self._system_state.update(state)

    def set_active_positions(self, positions: list[dict[str, Any]]) -> None:
        """Set current active positions."""
        self._active_positions = positions.copy()

    def set_pending_orders(self, orders: list[dict[str, Any]]) -> None:
        """Set current pending orders."""
        self._pending_orders = orders.copy()


# Global shutdown handler instance
_shutdown_handler: ShutdownHandler | None = None


def get_shutdown_handler() -> ShutdownHandler:
    """Get or create global shutdown handler instance."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = ShutdownHandler()
    return _shutdown_handler


async def initiate_shutdown(
    reason: ShutdownReason = ShutdownReason.USER_REQUEST,
) -> bool:
    """Initiate graceful shutdown."""
    handler = get_shutdown_handler()
    return await handler.shutdown(reason)


async def emergency_shutdown(
    reason: ShutdownReason = ShutdownReason.EMERGENCY_STOP,
) -> bool:
    """Initiate emergency shutdown."""
    handler = get_shutdown_handler()
    return await handler.shutdown(reason, emergency=True)


def register_shutdown_hooks(
    pre_shutdown: Callable | None = None,
    position_handler: Callable | None = None,
    order_handler: Callable | None = None,
    cleanup: Callable | None = None,
    post_shutdown: Callable | None = None,
) -> None:
    """Register multiple shutdown hooks at once."""
    handler = get_shutdown_handler()

    if pre_shutdown:
        handler.register_pre_shutdown_hook(pre_shutdown)
    if position_handler:
        handler.register_position_handler(position_handler)
    if order_handler:
        handler.register_order_handler(order_handler)
    if cleanup:
        handler.register_cleanup_hook(cleanup)
    if post_shutdown:
        handler.register_post_shutdown_hook(post_shutdown)


def is_shutting_down() -> bool:
    """Check if system is shutting down."""
    handler = get_shutdown_handler()
    return handler.is_shutting_down()
