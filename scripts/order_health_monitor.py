"""
Order Health Monitor - Enhanced monitoring and alerting for order management.

Addresses the critical issues:
- Orders stuck in "NEW" status without proper timeout handling
- Partial fills not being handled optimally  
- Need for comprehensive order status monitoring
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

# AI-AGENT-REF: Import configuration and execution modules
from ai_trading.config import management as config
from ai_trading.config.management import TradingConfig
CONFIG = TradingConfig()
from trade_execution import OrderInfo, _active_orders, _order_tracking_lock

logger = logging.getLogger(__name__)


@dataclass
class OrderHealthMetrics:
    """Comprehensive order health metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    canceled_orders: int = 0
    rejected_orders: int = 0
    stuck_orders: int = 0
    avg_fill_time: float = 0.0
    avg_fill_rate: float = 0.0
    success_rate: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PartialFillInfo:
    """Information about partial fills."""
    order_id: str
    symbol: str
    total_qty: int
    filled_qty: int
    remaining_qty: int
    fill_rate: float
    first_fill_time: datetime
    last_update: datetime
    retry_count: int = 0


class OrderHealthMonitor:
    """
    Comprehensive order health monitoring and alerting system.
    
    Addresses critical order management issues by providing:
    - Real-time order status monitoring
    - Automated timeout handling
    - Partial fill tracking and retry management
    - Health metrics and alerting
    """

    def __init__(self, execution_engine=None):
        """Initialize the order health monitor."""
        self.execution_engine = execution_engine
        self.logger = logging.getLogger(__name__ + ".OrderHealthMonitor")

        # Order tracking
        self._partial_fills: dict[str, PartialFillInfo] = {}
        self._fill_times: deque = deque(maxlen=1000)  # Recent fill times for metrics
        self._order_metrics: deque = deque(maxlen=100)  # Recent metrics snapshots

        # Configuration
        self.order_timeout_seconds = getattr(config, 'ORDER_TIMEOUT_SECONDS', 300)
        self.cleanup_interval = getattr(config, 'ORDER_STALE_CLEANUP_INTERVAL', 60)
        self.fill_rate_target = getattr(config, 'ORDER_FILL_RATE_TARGET', 0.80)
        self.max_retry_attempts = getattr(config, 'ORDER_MAX_RETRY_ATTEMPTS', 3)

        # Health monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Alerting thresholds
        self.alert_thresholds = {
            'success_rate_min': 0.70,  # Alert if success rate below 70%
            'avg_fill_time_max': 180,  # Alert if avg fill time > 3 minutes
            'stuck_orders_max': 5,     # Alert if > 5 orders stuck
            'partial_fill_rate_max': 0.30  # Alert if > 30% partial fills
        }

        self.logger.info("OrderHealthMonitor initialized with timeout=%ds, cleanup_interval=%ds",
                        self.order_timeout_seconds, self.cleanup_interval)

    def start_monitoring(self) -> None:
        """Start the order health monitoring thread."""
        if self._monitoring_active:
            self.logger.warning("Order monitoring already active")
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Order health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the order health monitoring thread."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info("Order health monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Check for stale orders
                self._check_stale_orders()

                # Update partial fill tracking
                self._update_partial_fill_tracking()

                # Generate health metrics
                metrics = self._calculate_health_metrics()

                # Check for alerts
                self._check_health_alerts(metrics)

                # Store metrics
                with self._lock:
                    self._order_metrics.append(metrics)

                # Sleep until next check
                time.sleep(self.cleanup_interval)

            except Exception as e:
                self.logger.error("Error in order monitoring loop: %s", e, exc_info=True)
                time.sleep(10)  # Back off on errors

    def _check_stale_orders(self) -> None:
        """Check for and handle stale orders."""
        current_time = time.time()
        stale_orders = []

        with _order_tracking_lock:
            for order_id, order_info in _active_orders.items():
                age = current_time - order_info.submitted_time
                if age > self.order_timeout_seconds and not order_info.cancel_attempted:
                    stale_orders.append(order_info)

        for order_info in stale_orders:
            self.logger.warning("STALE_ORDER_DETECTED", extra={
                "order_id": order_info.order_id,
                "symbol": order_info.symbol,
                "age_seconds": current_time - order_info.submitted_time,
                "status": order_info.last_status
            })

            # Attempt to cancel stale order
            if self.execution_engine:
                try:
                    success = self.execution_engine._cancel_stale_order(order_info.order_id)
                    if success:
                        self.logger.info("STALE_ORDER_CANCELED", extra={
                            "order_id": order_info.order_id,
                            "symbol": order_info.symbol
                        })
                except Exception as e:
                    self.logger.error("Failed to cancel stale order %s: %s", order_info.order_id, e)

    def _update_partial_fill_tracking(self) -> None:
        """Update tracking of partial fills and retry logic."""
        current_time = datetime.now(UTC)
        retry_candidates = []

        with self._lock:
            for order_id, partial_info in list(self._partial_fills.items()):
                # Check if partial fill is getting stale
                time_since_update = (current_time - partial_info.last_update).total_seconds()

                if time_since_update > 300:  # 5 minutes since last update
                    if partial_info.retry_count < self.max_retry_attempts:
                        retry_candidates.append(partial_info)
                    else:
                        # Give up on this partial fill
                        self.logger.warning("PARTIAL_FILL_ABANDONED", extra={
                            "order_id": order_id,
                            "symbol": partial_info.symbol,
                            "fill_rate": partial_info.fill_rate,
                            "retry_count": partial_info.retry_count
                        })
                        del self._partial_fills[order_id]

        # Process retry candidates
        for partial_info in retry_candidates:
            self._retry_partial_fill(partial_info)

    def _retry_partial_fill(self, partial_info: PartialFillInfo) -> None:
        """Attempt to retry completing a partial fill."""
        if not self.execution_engine:
            return

        self.logger.info("PARTIAL_FILL_RETRY_ATTEMPT", extra={
            "order_id": partial_info.order_id,
            "symbol": partial_info.symbol,
            "remaining_qty": partial_info.remaining_qty,
            "retry_count": partial_info.retry_count + 1
        })

        try:
            # This would trigger a new order for the remaining quantity
            # For now, just update the retry count and log
            with self._lock:
                if partial_info.order_id in self._partial_fills:
                    self._partial_fills[partial_info.order_id].retry_count += 1
                    self._partial_fills[partial_info.order_id].last_update = datetime.now(UTC)

        except Exception as e:
            self.logger.error("Partial fill retry failed for %s: %s", partial_info.order_id, e)

    def _calculate_health_metrics(self) -> OrderHealthMetrics:
        """Calculate comprehensive health metrics."""
        current_time = time.time()

        # Get current order statistics
        with _order_tracking_lock:
            active_orders = list(_active_orders.values())

        with self._lock:
            partial_fills = list(self._partial_fills.values())
            recent_fill_times = list(self._fill_times)

        # Calculate metrics
        total_active = len(active_orders)
        stuck_orders = sum(1 for order in active_orders
                          if (current_time - order.submitted_time) > self.order_timeout_seconds)

        # Calculate average fill time from recent data
        avg_fill_time = 0.0
        if recent_fill_times:
            avg_fill_time = sum(recent_fill_times) / len(recent_fill_times)

        # Calculate fill rates and success rates from recent metrics
        partial_fill_count = len(partial_fills)

        # Estimate success rate based on stuck vs active orders
        success_rate = 1.0
        if total_active > 0:
            success_rate = max(0.0, 1.0 - (stuck_orders / total_active))

        # Calculate average fill rate from partial fills
        avg_fill_rate = 1.0
        if partial_fills:
            avg_fill_rate = sum(pf.fill_rate for pf in partial_fills) / len(partial_fills)

        return OrderHealthMetrics(
            total_orders=total_active,
            filled_orders=max(0, total_active - stuck_orders - partial_fill_count),
            partial_fills=partial_fill_count,
            stuck_orders=stuck_orders,
            avg_fill_time=avg_fill_time,
            avg_fill_rate=avg_fill_rate,
            success_rate=success_rate
        )

    def _check_health_alerts(self, metrics: OrderHealthMetrics) -> None:
        """Check health metrics against alert thresholds."""
        alerts = []

        if metrics.success_rate < self.alert_thresholds['success_rate_min']:
            alerts.append(f"Low success rate: {metrics.success_rate:.1%}")

        if metrics.avg_fill_time > self.alert_thresholds['avg_fill_time_max']:
            alerts.append(f"High fill time: {metrics.avg_fill_time:.1f}s")

        if metrics.stuck_orders > self.alert_thresholds['stuck_orders_max']:
            alerts.append(f"Too many stuck orders: {metrics.stuck_orders}")

        if metrics.total_orders > 0:
            partial_rate = metrics.partial_fills / metrics.total_orders
            if partial_rate > self.alert_thresholds['partial_fill_rate_max']:
                alerts.append(f"High partial fill rate: {partial_rate:.1%}")

        if alerts:
            self.logger.error("ORDER_HEALTH_ALERTS", extra={
                "alerts": alerts,
                "metrics": {
                    "success_rate": metrics.success_rate,
                    "avg_fill_time": metrics.avg_fill_time,
                    "stuck_orders": metrics.stuck_orders,
                    "partial_fills": metrics.partial_fills,
                    "total_orders": metrics.total_orders
                }
            })

    def record_order_fill(self, order_id: str, fill_time: float, is_partial: bool = False,
                         fill_qty: int = 0, total_qty: int = 0) -> None:
        """Record an order fill for metrics tracking."""
        with self._lock:
            self._fill_times.append(fill_time)

        if is_partial and total_qty > 0:
            self._record_partial_fill(order_id, fill_qty, total_qty)

        self.logger.info("ORDER_FILL_RECORDED", extra={
            "order_id": order_id,
            "fill_time": fill_time,
            "is_partial": is_partial,
            "fill_rate": fill_qty / total_qty if total_qty > 0 else 1.0
        })

    def _record_partial_fill(self, order_id: str, filled_qty: int, total_qty: int) -> None:
        """Record a partial fill for tracking."""
        current_time = datetime.now(UTC)
        fill_rate = filled_qty / total_qty if total_qty > 0 else 0.0

        with self._lock:
            if order_id in self._partial_fills:
                # Update existing partial fill
                partial_info = self._partial_fills[order_id]
                partial_info.filled_qty = filled_qty
                partial_info.remaining_qty = total_qty - filled_qty
                partial_info.fill_rate = fill_rate
                partial_info.last_update = current_time
            else:
                # Create new partial fill record
                # We'd need additional context to get symbol and side
                # For now, create minimal record
                self._partial_fills[order_id] = PartialFillInfo(
                    order_id=order_id,
                    symbol="UNKNOWN",  # Would need to be passed in
                    total_qty=total_qty,
                    filled_qty=filled_qty,
                    remaining_qty=total_qty - filled_qty,
                    fill_rate=fill_rate,
                    first_fill_time=current_time,
                    last_update=current_time
                )

    def get_health_summary(self) -> dict:
        """Get a summary of current order health."""
        metrics = self._calculate_health_metrics()

        with self._lock:
            recent_metrics = list(self._order_metrics)[-10:] if self._order_metrics else []

        # Calculate trends
        trend_data = {}
        if len(recent_metrics) >= 2:
            latest = recent_metrics[-1]
            previous = recent_metrics[-2]

            trend_data = {
                "success_rate_trend": latest.success_rate - previous.success_rate,
                "fill_time_trend": latest.avg_fill_time - previous.avg_fill_time,
                "stuck_orders_trend": latest.stuck_orders - previous.stuck_orders
            }

        return {
            "current_metrics": {
                "total_orders": metrics.total_orders,
                "success_rate": round(metrics.success_rate, 3),
                "avg_fill_time": round(metrics.avg_fill_time, 1),
                "stuck_orders": metrics.stuck_orders,
                "partial_fills": metrics.partial_fills,
                "avg_fill_rate": round(metrics.avg_fill_rate, 3)
            },
            "trends": trend_data,
            "alerts_enabled": True,
            "monitoring_active": self._monitoring_active,
            "timestamp": datetime.now(UTC).isoformat()
        }

    def export_metrics(self, filepath: str) -> None:
        """Export health metrics to a file."""
        try:
            with self._lock:
                metrics_data = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "total_orders": m.total_orders,
                        "success_rate": m.success_rate,
                        "avg_fill_time": m.avg_fill_time,
                        "stuck_orders": m.stuck_orders,
                        "partial_fills": m.partial_fills,
                        "avg_fill_rate": m.avg_fill_rate
                    }
                    for m in self._order_metrics
                ]

            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            self.logger.info("Order health metrics exported to %s", filepath)

        except Exception as e:
            self.logger.error("Failed to export metrics: %s", e)


# Global instance for easy access
_order_health_monitor = None


def get_order_health_monitor(execution_engine=None) -> OrderHealthMonitor:
    """Get or create the global order health monitor instance."""
    global _order_health_monitor

    if _order_health_monitor is None:
        _order_health_monitor = OrderHealthMonitor(execution_engine)

    return _order_health_monitor


def start_order_monitoring(execution_engine=None) -> None:
    """Start order health monitoring."""
    monitor = get_order_health_monitor(execution_engine)
    monitor.start_monitoring()


def stop_order_monitoring() -> None:
    """Stop order health monitoring."""
    if _order_health_monitor:
        _order_health_monitor.stop_monitoring()


def get_order_health_summary() -> dict:
    """Get current order health summary."""
    if _order_health_monitor:
        return _order_health_monitor.get_health_summary()
    else:
        return {"error": "Order health monitoring not initialized"}
