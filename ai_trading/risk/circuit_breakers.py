from __future__ import annotations
# ruff: noqa

"""Circuit breakers and safety mechanisms for production trading."""

import functools
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger
from json import JSONDecodeError

# Consistent exception tuple without hard dependency on requests
try:  # pragma: no cover
    import requests  # type: ignore
    RequestException = requests.exceptions.RequestException  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover  # AI-AGENT-REF: narrow requests import
    class RequestException(Exception):
        pass
COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)

from ..core.constants import PERFORMANCE_THRESHOLDS


class CircuitBreaker:
    """Simple decorator-friendly circuit breaker."""  # AI-AGENT-REF

    def __init__(self, *args, **kwargs):  # noqa: D401 - minimal stub
        pass

    def call(self, fn):
        @functools.wraps(fn)
        def _wrapped(*a, **k):
            return fn(*a, **k)

        return _wrapped

    def __call__(self, fn):
        return self.call(fn)


DEFAULT_BREAKER = CircuitBreaker()


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Limited operation


class SafetyLevel(Enum):
    """Safety alert level enumeration."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DrawdownCircuitBreaker:
    """
    Drawdown-based circuit breaker for portfolio protection.

    Monitors portfolio drawdown and automatically halts trading
    when predefined thresholds are exceeded.
    """

    def __init__(self, max_drawdown: float = None, recovery_threshold: float = 0.8):
        """Initialize drawdown circuit breaker."""
        # AI-AGENT-REF: Drawdown-based circuit breaker for safety
        self.max_drawdown = max_drawdown or PERFORMANCE_THRESHOLDS["MAX_DRAWDOWN"]
        self.recovery_threshold = recovery_threshold  # 80% recovery before reset

        self.state = CircuitBreakerState.CLOSED
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.halt_timestamp = None
        self.reset_callbacks = []

        logger.info(
            f"DrawdownCircuitBreaker initialized with max_drawdown={self._safe_format_percentage(self.max_drawdown)}"
        )

    def _safe_format_percentage(self, value) -> str:
        """
        Safely format a value as a percentage, handling MagicMock objects during testing.

        AI-AGENT-REF: Defensive programming for test compatibility
        """
        try:
            if hasattr(value, "_mock_name"):  # Check if it's a MagicMock
                return f"<Mock: {value._mock_name or 'percentage'}>"
            return f"{value:.2%}"
        except (TypeError, ValueError, AttributeError):
            return f"<{type(value).__name__}: {value}>"

    def update_equity(self, current_equity: float) -> bool:
        """
        Update current equity and check circuit breaker status.

        Args:
            current_equity: Current portfolio equity value

        Returns:
            True if trading is allowed, False if halted
        """
        try:
            # AI-AGENT-REF: Add input validation for edge cases
            if current_equity is None or not isinstance(current_equity, int | float):
                logger.warning(
                    f"Invalid equity value: {current_equity} (type: {type(current_equity)})"
                )
                return False

            # Handle negative equity (unlikely but possible with margin)
            if current_equity < 0:
                logger.warning(f"Negative equity detected: {current_equity}")
                return False

            # Update peak equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity

            # Calculate current drawdown
            if self.peak_equity > 0:
                self.current_drawdown = (
                    self.peak_equity - current_equity
                ) / self.peak_equity
            else:
                self.current_drawdown = 0.0

            # AI-AGENT-REF: Add bounds checking for drawdown calculation
            if self.current_drawdown < 0:
                logger.debug(
                    f"Negative drawdown detected: {self.current_drawdown} (equity increased above peak)"
                )
                self.current_drawdown = 0.0

            # Check for circuit breaker trigger
            if self.state == CircuitBreakerState.CLOSED:
                if self.current_drawdown >= self.max_drawdown:
                    self._trigger_halt("Maximum drawdown exceeded")
                    return False

            # Check for recovery
            elif self.state == CircuitBreakerState.OPEN:
                recovery_ratio = (
                    current_equity / self.peak_equity if self.peak_equity > 0 else 0
                )
                if recovery_ratio >= self.recovery_threshold:
                    self._reset_breaker("Drawdown recovery achieved")
                    return True
                return False

            return self.state == CircuitBreakerState.CLOSED

        except COMMON_EXC as e:
            logger.error(f"Error updating drawdown circuit breaker: {e}", exc_info=True)
            # AI-AGENT-REF: Return False for safety when circuit breaker fails
            return False

    def _trigger_halt(self, reason: str):
        """Trigger circuit breaker halt."""
        try:
            self.state = CircuitBreakerState.OPEN
            self.halt_timestamp = datetime.now(UTC)

            logger.critical(
                f"TRADING HALTED - Drawdown Circuit Breaker: {reason}. "
                f"Current drawdown: {self._safe_format_percentage(self.current_drawdown)}"
            )

            # Execute callbacks
            for callback in self.reset_callbacks:
                try:
                    callback(
                        "halt",
                        {
                            "reason": reason,
                            "drawdown": self.current_drawdown,
                            "timestamp": self.halt_timestamp,
                        },
                    )
                except COMMON_EXC as e:
                    logger.error(f"Error in circuit breaker callback: {e}")

        except COMMON_EXC as e:
            logger.error(f"Error triggering drawdown halt: {e}")

    def _reset_breaker(self, reason: str):
        """Reset circuit breaker to normal operation."""
        try:
            self.state = CircuitBreakerState.CLOSED
            self.halt_timestamp = None

            logger.info(f"TRADING RESUMED - Drawdown Circuit Breaker: {reason}")

            # Execute callbacks
            for callback in self.reset_callbacks:
                try:
                    callback(
                        "reset",
                        {
                            "reason": reason,
                            "drawdown": self.current_drawdown,
                            "timestamp": datetime.now(UTC),
                        },
                    )
                except COMMON_EXC as e:
                    logger.error(f"Error in circuit breaker callback: {e}")

        except COMMON_EXC as e:
            logger.error(f"Error resetting drawdown breaker: {e}")

    def add_callback(self, callback: Callable[[str, dict], None]):
        """Add callback for circuit breaker events."""
        self.reset_callbacks.append(callback)

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "peak_equity": self.peak_equity,
            "halt_timestamp": self.halt_timestamp,
            "trading_allowed": self.state == CircuitBreakerState.CLOSED,
        }


class VolatilityCircuitBreaker:
    """
    Volatility-based circuit breaker for extreme market conditions.

    Monitors market volatility and automatically reduces or halts
    trading during periods of extreme volatility.
    """

    def __init__(
        self, high_vol_threshold: float = 0.5, extreme_vol_threshold: float = 1.0
    ):
        """Initialize volatility circuit breaker."""
        # AI-AGENT-REF: Volatility-based circuit breaker
        self.high_vol_threshold = high_vol_threshold  # 50% volatility - reduce trading
        self.extreme_vol_threshold = (
            extreme_vol_threshold  # 100% volatility - halt trading
        )

        self.state = CircuitBreakerState.CLOSED
        self.current_volatility = 0.0
        self.position_size_multiplier = 1.0
        self.last_volatility_update = datetime.now(UTC)

        logger.info(
            f"VolatilityCircuitBreaker initialized with thresholds: "
            f"high={self._safe_format_percentage(high_vol_threshold)}, extreme={self._safe_format_percentage(extreme_vol_threshold)}"
        )

    def _safe_format_percentage(self, value) -> str:
        """
        Safely format a value as a percentage, handling MagicMock objects during testing.

        AI-AGENT-REF: Defensive programming for test compatibility
        """
        try:
            if hasattr(value, "_mock_name"):  # Check if it's a MagicMock
                return f"<Mock: {value._mock_name or 'percentage'}>"
            return f"{value:.1%}"
        except (TypeError, ValueError, AttributeError):
            return f"<{type(value).__name__}: {value}>"

    def update_volatility(self, volatility: float) -> dict[str, Any]:
        """
        Update current volatility and adjust trading parameters.

        Args:
            volatility: Current annualized volatility (e.g., 0.2 for 20%)

        Returns:
            Dictionary with trading adjustments
        """
        try:
            self.current_volatility = volatility
            self.last_volatility_update = datetime.now(UTC)

            # Determine state and adjustments based on volatility
            if volatility >= self.extreme_vol_threshold:
                self.state = CircuitBreakerState.OPEN
                self.position_size_multiplier = 0.0
                status = "EXTREME_VOLATILITY_HALT"
                logger.critical(
                    f"EXTREME VOLATILITY DETECTED: {self._safe_format_percentage(volatility)} - Trading halted"
                )

            elif volatility >= self.high_vol_threshold:
                self.state = CircuitBreakerState.HALF_OPEN
                # Reduce position sizes proportionally
                reduction_factor = (volatility - self.high_vol_threshold) / (
                    self.extreme_vol_threshold - self.high_vol_threshold
                )
                self.position_size_multiplier = max(0.1, 1.0 - reduction_factor * 0.8)
                status = "HIGH_VOLATILITY_REDUCTION"
                logger.warning(
                    f"HIGH VOLATILITY: {self._safe_format_percentage(volatility)} - Position sizes reduced to {self._safe_format_percentage(self.position_size_multiplier)}"
                )

            else:
                self.state = CircuitBreakerState.CLOSED
                self.position_size_multiplier = 1.0
                status = "NORMAL_OPERATION"
                logger.debug(
                    f"Normal volatility: {self._safe_format_percentage(volatility)}"
                )

            return {
                "status": status,
                "trading_allowed": self.state != CircuitBreakerState.OPEN,
                "position_size_multiplier": self.position_size_multiplier,
                "volatility": volatility,
                "state": self.state.value,
            }

        except COMMON_EXC as e:
            logger.error(f"Error updating volatility circuit breaker: {e}")
            return {
                "status": "ERROR",
                "trading_allowed": False,
                "position_size_multiplier": 0.0,
                "volatility": 0.0,
                "state": "error",
            }

    def get_status(self) -> dict[str, Any]:
        """Get current volatility circuit breaker status."""
        return {
            "state": self.state.value,
            "current_volatility": self.current_volatility,
            "high_vol_threshold": self.high_vol_threshold,
            "extreme_vol_threshold": self.extreme_vol_threshold,
            "position_size_multiplier": self.position_size_multiplier,
            "last_update": self.last_volatility_update,
            "trading_allowed": self.state != CircuitBreakerState.OPEN,
        }


class TradingHaltManager:
    """
    Comprehensive trading halt management system.

    Coordinates multiple circuit breakers and safety mechanisms
    to provide unified trading halt control.
    """

    def __init__(self):
        """Initialize trading halt manager."""
        # AI-AGENT-REF: Comprehensive trading halt management
        self.drawdown_breaker = DrawdownCircuitBreaker()
        self.volatility_breaker = VolatilityCircuitBreaker()

        # Manual halt controls
        self.manual_halt = False
        self.manual_halt_reason = ""
        self.manual_halt_timestamp = None

        # Safety counters
        self.daily_trade_count = 0
        self.daily_loss_amount = 0.0
        self.max_daily_trades = 1000  # From SYSTEM_LIMITS
        self.max_daily_loss = 0.05  # 5% max daily loss

        # Emergency controls
        self.emergency_stop = False
        self.emergency_callbacks = []

        # Thread safety
        self._lock = threading.RLock()

        logger.info("TradingHaltManager initialized")

    def _safe_format_percentage(self, value) -> str:
        """
        Safely format a value as a percentage, handling MagicMock objects during testing.

        AI-AGENT-REF: Defensive programming for test compatibility
        """
        try:
            if hasattr(value, "_mock_name"):  # Check if it's a MagicMock
                return f"<Mock: {value._mock_name or 'percentage'}>"
            return f"{value:.2%}"
        except (TypeError, ValueError, AttributeError):
            return f"<{type(value).__name__}: {value}>"

    def is_trading_allowed(self) -> dict[str, Any]:
        """
        Check if trading is currently allowed.

        Returns:
            Dictionary with trading status and reasons
        """
        try:
            with self._lock:
                status = {
                    "trading_allowed": True,
                    "reasons": [],
                    "position_size_multiplier": 1.0,
                    "circuit_breakers": {},
                }

                # Check manual halt
                if self.manual_halt:
                    status["trading_allowed"] = False
                    status["reasons"].append(f"Manual halt: {self.manual_halt_reason}")

                # Check emergency stop
                if self.emergency_stop:
                    status["trading_allowed"] = False
                    status["reasons"].append("Emergency stop activated")

                # Check drawdown circuit breaker
                drawdown_status = self.drawdown_breaker.get_status()
                status["circuit_breakers"]["drawdown"] = drawdown_status
                if not drawdown_status["trading_allowed"]:
                    status["trading_allowed"] = False
                    status["reasons"].append(
                        f"Drawdown limit exceeded: {self._safe_format_percentage(drawdown_status['current_drawdown'])}"
                    )

                # Check volatility circuit breaker
                volatility_status = self.volatility_breaker.get_status()
                status["circuit_breakers"]["volatility"] = volatility_status
                if not volatility_status["trading_allowed"]:
                    status["trading_allowed"] = False
                    status["reasons"].append(
                        f"Extreme volatility: {self._safe_format_percentage(volatility_status['current_volatility'])}"
                    )
                else:
                    # Apply volatility position size reduction
                    status["position_size_multiplier"] *= volatility_status[
                        "position_size_multiplier"
                    ]

                # Check daily limits
                if self.daily_trade_count >= self.max_daily_trades:
                    status["trading_allowed"] = False
                    status["reasons"].append(
                        f"Daily trade limit exceeded: {self.daily_trade_count}"
                    )

                if self.daily_loss_amount >= self.max_daily_loss:
                    status["trading_allowed"] = False
                    status["reasons"].append(
                        f"Daily loss limit exceeded: {self._safe_format_percentage(self.daily_loss_amount)}"
                    )

                return status

        except COMMON_EXC as e:
            logger.error(f"Error checking trading status: {e}")
            return {
                "trading_allowed": False,
                "reasons": [f"System error: {e}"],
                "position_size_multiplier": 0.0,
                "circuit_breakers": {},
            }

    def update_equity(self, current_equity: float):
        """Update equity for drawdown monitoring."""
        try:
            with self._lock:
                self.drawdown_breaker.update_equity(current_equity)
        except COMMON_EXC as e:
            logger.error(f"Error updating equity in halt manager: {e}")

    def update_volatility(self, volatility: float):
        """Update volatility for volatility monitoring."""
        try:
            with self._lock:
                return self.volatility_breaker.update_volatility(volatility)
        except COMMON_EXC as e:
            logger.error(f"Error updating volatility in halt manager: {e}")
            return {"status": "ERROR", "trading_allowed": False}

    def manual_halt_trading(self, reason: str):
        """Manually halt trading with reason."""
        try:
            with self._lock:
                self.manual_halt = True
                self.manual_halt_reason = reason
                self.manual_halt_timestamp = datetime.now(UTC)

                logger.critical(f"MANUAL TRADING HALT: {reason}")

        except COMMON_EXC as e:
            logger.error(f"Error setting manual halt: {e}")

    def resume_trading(self, reason: str = "Manual resume"):
        """Resume trading from manual halt."""
        try:
            with self._lock:
                self.manual_halt = False
                self.manual_halt_reason = ""
                self.manual_halt_timestamp = None

                logger.info(f"TRADING RESUMED: {reason}")

        except COMMON_EXC as e:
            logger.error(f"Error resuming trading: {e}")

    def emergency_stop_all(self, reason: str = "Emergency stop"):
        """Activate emergency stop - highest priority halt."""
        try:
            with self._lock:
                self.emergency_stop = True

                logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

                # Execute emergency callbacks
                for callback in self.emergency_callbacks:
                    try:
                        callback(reason)
                    except COMMON_EXC as e:
                        logger.error(f"Error in emergency callback: {e}")

        except COMMON_EXC as e:
            logger.error(f"Error activating emergency stop: {e}")

    def reset_emergency_stop(self, reason: str = "Manual reset"):
        """Reset emergency stop (requires manual intervention)."""
        try:
            with self._lock:
                self.emergency_stop = False

                logger.info(f"EMERGENCY STOP RESET: {reason}")

        except COMMON_EXC as e:
            logger.error(f"Error resetting emergency stop: {e}")

    def record_trade(self, trade_pnl: float = 0.0):
        """Record a trade for daily limit tracking."""
        try:
            with self._lock:
                self.daily_trade_count += 1
                if trade_pnl < 0:
                    self.daily_loss_amount += abs(trade_pnl)

                logger.debug(
                    f"Trade recorded: count={self.daily_trade_count}, "
                    f"daily_loss={self.daily_loss_amount:.4f}"
                )

        except COMMON_EXC as e:
            logger.error(f"Error recording trade: {e}")

    def reset_daily_counters(self):
        """Reset daily counters (typically called at market open)."""
        try:
            with self._lock:
                self.daily_trade_count = 0
                self.daily_loss_amount = 0.0

                logger.info("Daily trading counters reset")

        except COMMON_EXC as e:
            logger.error(f"Error resetting daily counters: {e}")

    def add_emergency_callback(self, callback: Callable[[str], None]):
        """Add callback for emergency stop events."""
        self.emergency_callbacks.append(callback)

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive status of all safety systems."""
        try:
            with self._lock:
                trading_status = self.is_trading_allowed()

                return {
                    "timestamp": datetime.now(UTC),
                    "trading_status": trading_status,
                    "manual_controls": {
                        "manual_halt": self.manual_halt,
                        "manual_halt_reason": self.manual_halt_reason,
                        "manual_halt_timestamp": self.manual_halt_timestamp,
                        "emergency_stop": self.emergency_stop,
                    },
                    "daily_limits": {
                        "trade_count": self.daily_trade_count,
                        "max_daily_trades": self.max_daily_trades,
                        "daily_loss_amount": self.daily_loss_amount,
                        "max_daily_loss": self.max_daily_loss,
                    },
                    "circuit_breakers": trading_status["circuit_breakers"],
                }

        except COMMON_EXC as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return {"error": str(e)}


class DeadMansSwitch:
    """
    Dead man's switch for automated system monitoring.

    Requires periodic heartbeat signals and automatically triggers
    emergency procedures if heartbeat is not received within timeout.
    """

    def __init__(self, timeout_seconds: int = 300):  # 5 minute default
        """Initialize dead man's switch."""
        # AI-AGENT-REF: Dead man's switch for system monitoring
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = datetime.now(UTC)
        self.is_active = False
        self.emergency_callbacks = []
        self.monitoring_thread = None
        self._stop_event = threading.Event()

        logger.info(f"DeadMansSwitch initialized with timeout={timeout_seconds}s")

    def start_monitoring(self):
        """Start the dead man's switch monitoring."""
        try:
            if self.is_active:
                logger.warning("Dead man's switch already active")
                return

            self.is_active = True
            self.last_heartbeat = datetime.now(UTC)
            self._stop_event.clear()

            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True, name="DeadMansSwitch"
            )
            self.monitoring_thread.start()

            logger.info("Dead man's switch monitoring started")

        except COMMON_EXC as e:
            logger.error(f"Error starting dead man's switch: {e}")

    def stop_monitoring(self):
        """Stop the dead man's switch monitoring."""
        try:
            if not self.is_active:
                return

            self.is_active = False
            self._stop_event.set()

            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)

            logger.info("Dead man's switch monitoring stopped")

        except COMMON_EXC as e:
            logger.error(f"Error stopping dead man's switch: {e}")

    def heartbeat(self):
        """Send heartbeat signal to reset the timer."""
        try:
            if self.is_active:
                self.last_heartbeat = datetime.now(UTC)
                logger.debug("Dead man's switch heartbeat received")

        except COMMON_EXC as e:
            logger.error(f"Error processing heartbeat: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        try:
            while self.is_active and not self._stop_event.is_set():
                current_time = datetime.now(UTC)
                time_since_heartbeat = (
                    current_time - self.last_heartbeat
                ).total_seconds()

                if time_since_heartbeat > self.timeout_seconds:
                    logger.critical(
                        f"DEAD MAN'S SWITCH TRIGGERED - No heartbeat for {time_since_heartbeat:.0f}s"
                    )
                    self._trigger_emergency()
                    break

                # Check every 10 seconds
                self._stop_event.wait(10.0)

        except COMMON_EXC as e:
            logger.error(f"Error in dead man's switch monitoring loop: {e}")

    def _trigger_emergency(self):
        """Trigger emergency procedures."""
        try:
            # Execute all emergency callbacks
            for callback in self.emergency_callbacks:
                try:
                    callback("Dead man's switch timeout")
                except COMMON_EXC as e:
                    logger.error(f"Error in dead man's switch callback: {e}")

            # Stop monitoring after triggering
            self.is_active = False

        except COMMON_EXC as e:
            logger.error(f"Error triggering dead man's switch emergency: {e}")

    def add_emergency_callback(self, callback: Callable[[str], None]):
        """Add callback for emergency trigger events."""
        self.emergency_callbacks.append(callback)

    def get_status(self) -> dict[str, Any]:
        """Get current dead man's switch status."""
        try:
            current_time = datetime.now(UTC)
            time_since_heartbeat = (current_time - self.last_heartbeat).total_seconds()

            return {
                "is_active": self.is_active,
                "last_heartbeat": self.last_heartbeat,
                "time_since_heartbeat": time_since_heartbeat,
                "timeout_seconds": self.timeout_seconds,
                "status": (
                    "OK" if time_since_heartbeat < self.timeout_seconds else "TIMEOUT"
                ),
                "monitoring_thread_alive": (
                    self.monitoring_thread.is_alive()
                    if self.monitoring_thread
                    else False
                ),
            }

        except COMMON_EXC as e:
            logger.error(f"Error getting dead man's switch status: {e}")
            return {"error": str(e)}
