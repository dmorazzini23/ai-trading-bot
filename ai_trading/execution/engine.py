"""
Core execution engine for institutional order management.

Provides order lifecycle management, execution algorithms,
and real-time execution monitoring with institutional controls.
"""
from __future__ import annotations
from ai_trading.logging import get_logger
import math
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any
try:  # pragma: no cover - optional dependency
    from alpaca_trade_api.rest import APIError  # type: ignore
except Exception:  # pragma: no cover - fallback when SDK missing
    class APIError(Exception):
        """Fallback APIError when alpaca-trade-api is unavailable."""

        pass
from ai_trading.logging.emit_once import emit_once

logger = get_logger(__name__)
ORDER_STALE_AFTER_S = 8 * 60

from ai_trading.monitoring.order_health_monitor import (
    OrderInfo,
    _active_orders as _mon_active,
    _order_tracking_lock as _mon_lock,
)
_active_orders = _mon_active
_order_tracking_lock = _mon_lock

def _cleanup_stale_orders(now: float | None=None, max_age_s: int | None=None) -> int:
    """Remove orders older than ``max_age_s`` and return count."""
    max_age = max_age_s if max_age_s is not None else ORDER_STALE_AFTER_S
    now_s = now if now is not None else time.time()
    removed = 0
    with _order_tracking_lock:
        for oid, info in list(_active_orders.items()):
            if now_s - info.submitted_time >= max_age:
                _active_orders.pop(oid, None)
                removed += 1
    return removed
from ai_trading.market.symbol_specs import TICK_BY_SYMBOL, get_lot_size, get_tick_size
from ai_trading.math.money import Money, round_to_lot, round_to_tick
from ..core.constants import EXECUTION_PARAMETERS
from ..core.enums import OrderSide, OrderStatus, OrderType
from .idempotency import OrderIdempotencyCache

def _ensure_positive_qty(qty: float) -> float:
    if qty is None:
        raise ValueError('qty_none')
    q = float(qty)
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError(f'invalid_qty:{qty}')
    return q

def _ensure_valid_price(price: float | None) -> float | None:
    if price is None:
        return None
    p = float(price)
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError(f'invalid_price:{price}')
    return p

class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    MARKET = 'market'
    LIMIT = 'limit'
    VWAP = 'vwap'
    TWAP = 'twap'
    IMPLEMENTATION_SHORTFALL = 'implementation_shortfall'
    ICEBERG = 'iceberg'

class Order:
    """
    Order representation for institutional execution.

    Comprehensive order model with execution tracking,
    partial fills, and institutional metadata.
    """

    def __init__(self, symbol: str, side: OrderSide, quantity: int, order_type: OrderType=OrderType.MARKET, price: Money=None, **kwargs):
        """Initialize order with institutional parameters."""
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        tick = TICK_BY_SYMBOL.get(symbol)
        self.price = Money(price, tick) if price is not None else None
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.average_fill_price = Money(0)
        self.fills = []
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self.executed_at = None
        self.client_order_id = kwargs.get('client_order_id', f'ord_{int(time.time())}')
        self.strategy_id = kwargs.get('strategy_id')
        self.execution_algorithm = kwargs.get('execution_algorithm', ExecutionAlgorithm.MARKET)
        self.time_in_force = kwargs.get('time_in_force', 'DAY')
        self.min_quantity = kwargs.get('min_quantity', 0)
        self.stop_price = kwargs.get('stop_price')
        self.target_price = kwargs.get('target_price')
        self.max_participation_rate = kwargs.get('max_participation_rate', 0.1)
        self.max_slippage_bps = kwargs.get('max_slippage_bps', EXECUTION_PARAMETERS['MAX_SLIPPAGE_BPS'])
        self.urgency_level = kwargs.get('urgency_level', 'normal')
        self.notes = kwargs.get('notes', '')
        self.source_system = kwargs.get('source_system', 'ai_trading')
        self.parent_order_id = kwargs.get('parent_order_id')
        logger.debug(f'Order created: {self.id} {self.side} {self.quantity} {self.symbol}')

    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        return self.filled_quantity / self.quantity * 100 if self.quantity > 0 else 0

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

    def add_fill(self, quantity: int, price: Money, timestamp: datetime=None):
        """Add a fill to the order with precise money math."""
        if timestamp is None:
            timestamp = datetime.now(UTC)
        tick = TICK_BY_SYMBOL.get(self.symbol)
        if not isinstance(price, Money):
            price = Money(price, tick)
        fill = {'quantity': quantity, 'price': price, 'timestamp': timestamp, 'fill_id': str(uuid.uuid4())}
        self.fills.append(fill)
        self.filled_quantity += quantity
        total_value = sum((Money(f['quantity']) * f['price'] for f in self.fills))
        self.average_fill_price = total_value / Money(self.filled_quantity) if self.filled_quantity > 0 else Money(0)
        if self.is_filled:
            self.status = OrderStatus.FILLED
            self.executed_at = timestamp
        elif self.is_partially_filled:
            self.status = OrderStatus.PARTIALLY_FILLED
        self.updated_at = timestamp
        logger.debug(f'Fill added to order {self.id}: {quantity}@{price} ({self.fill_percentage:.1f}% filled)')

    def cancel(self, reason: str='User cancelled'):
        """Cancel the order."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            logger.warning(f'Cannot cancel order {self.id} in status {self.status}')
            return False
        self.status = OrderStatus.CANCELED
        self.updated_at = datetime.now(UTC)
        self.notes += f' | Cancelled: {reason}'
        logger.info(f'Order {self.id} cancelled: {reason}')
        return True

    def to_dict(self) -> dict:
        """Convert order to dictionary representation."""
        return {'id': self.id, 'symbol': self.symbol, 'side': self.side.value if isinstance(self.side, OrderSide) else self.side, 'quantity': self.quantity, 'order_type': self.order_type.value if isinstance(self.order_type, OrderType) else self.order_type, 'price': self.price, 'status': self.status.value if isinstance(self.status, OrderStatus) else self.status, 'filled_quantity': self.filled_quantity, 'average_fill_price': self.average_fill_price, 'created_at': self.created_at.isoformat(), 'updated_at': self.updated_at.isoformat(), 'executed_at': self.executed_at.isoformat() if self.executed_at else None, 'client_order_id': self.client_order_id, 'strategy_id': self.strategy_id, 'fills': self.fills, 'notional_value': self.notional_value, 'fill_percentage': self.fill_percentage}

class OrderManager:
    """
    Order lifecycle management for institutional execution.

    Manages order routing, execution tracking, and provides
    real-time order monitoring with institutional controls.
    """

    def __init__(self):
        """Initialize order manager."""
        self.orders: dict[str, Order] = {}
        self.active_orders: dict[str, Order] = {}
        self.execution_callbacks: list[Callable] = []
        self.max_concurrent_orders = EXECUTION_PARAMETERS.get('MAX_CONCURRENT_ORDERS', 100)
        self.order_timeout = EXECUTION_PARAMETERS.get('ORDER_TIMEOUT_SECONDS', 300)
        self.retry_attempts = EXECUTION_PARAMETERS.get('RETRY_ATTEMPTS', 3)
        self._monitor_thread = None
        self._monitor_running = False
        self._idempotency_cache: OrderIdempotencyCache | None = None
        emit_once(logger, 'ORDER_MANAGER_INIT', 'info', 'OrderManager initialized')

    def _ensure_idempotency_cache(self) -> OrderIdempotencyCache:
        """Ensure idempotency cache is instantiated."""
        if self._idempotency_cache is None:
            try:
                self._idempotency_cache = OrderIdempotencyCache()
            except (KeyError, ValueError, TypeError, RuntimeError) as e:
                logger.error('IDEMPOTENCY_CACHE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
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
            if not self._validate_order(order):
                return False
            cache = self._ensure_idempotency_cache()
            key = cache.generate_key(order.symbol, order.side, order.quantity, datetime.now(UTC))
            if cache.is_duplicate(key):
                logger.warning(f'ORDER_DUPLICATE_SKIPPED: {order.symbol} {order.side} {order.quantity}')
                order.status = OrderStatus.REJECTED
                order.notes += ' | Rejected: Duplicate order detected'
                return False
            if len(self.active_orders) >= self.max_concurrent_orders:
                logger.error(f'Cannot submit order: max concurrent orders reached ({self.max_concurrent_orders})')
                order.status = OrderStatus.REJECTED
                order.notes += ' | Rejected: Max concurrent orders reached'
                return False
            self.orders[order.id] = order
            self.active_orders[order.id] = order
            cache.mark_submitted(key, order.id)
            if not self._monitor_running:
                self.start_monitoring()
            logger.info(f'Order submitted: {order.id} {order.side} {order.quantity} {order.symbol}')
            self._notify_callbacks(order, 'submitted')
            return True
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_API_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'op': 'submit', 'symbol': order.symbol, 'qty': order.quantity, 'side': getattr(order.side, 'value', order.side), 'type': getattr(order.order_type, 'value', order.order_type)})
            order.status = OrderStatus.REJECTED
            order.notes += f' | Error: {e}'
            return False

    def cancel_order(self, order_id: str, reason: str='User request') -> bool:
        """Cancel an active order."""
        if not order_id:
            logger.warning('CANCEL_SKIPPED', extra={'reason': 'empty_order_id'})
            return False
        try:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning(f'Cannot cancel order {order_id}: not found in active orders')
                return False
            success = order.cancel(reason)
            if success:
                self.active_orders.pop(order_id, None)
                self._notify_callbacks(order, 'cancelled')
            return success
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_API_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'op': 'cancel', 'order_id': order_id})
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

    def get_order_history(self, symbol: str=None, limit: int=100) -> list[dict]:
        """Get order history with optional filtering."""
        orders = list(self.orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
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
        logger.info('Order monitoring started')

    def stop_monitoring(self):
        """Stop order monitoring thread."""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info('Order monitoring stopped')

    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission."""
        try:
            if not order.symbol or order.quantity <= 0:
                logger.error(f'Invalid order parameters: symbol={order.symbol}, quantity={order.quantity}')
                return False
            tick = get_tick_size(order.symbol)
            lot = get_lot_size(order.symbol)
            original_quantity = order.quantity
            order.quantity = round_to_lot(order.quantity, lot)
            if original_quantity != order.quantity:
                logger.debug(f'Quantity adjusted for {order.symbol}: {original_quantity} -> {order.quantity} (lot={lot})')
            if order.order_type == OrderType.LIMIT:
                if not order.price or order.price <= 0:
                    logger.error(f'Limit order requires valid price: {order.price}')
                    return False
                if not isinstance(order.price, Money):
                    order.price = Money(order.price)
                original_price = order.price
                order.price = round_to_tick(order.price, tick)
                if float(original_price) != float(order.price):
                    logger.debug(f'Price adjusted for {order.symbol}: {original_price} -> {order.price} (tick={tick})')
            if order.side not in [OrderSide.BUY, OrderSide.SELL]:
                logger.error(f'Invalid order side: {order.side}')
                return False
            return True
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.error('ORDER_VALIDATION_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return False

    def _monitor_orders(self):
        """Monitor active orders for timeouts and updates."""
        while self._monitor_running:
            try:
                current_time = datetime.now(UTC)
                expired_orders = []
                for order_id, order in list(self.active_orders.items()):
                    age_seconds = (current_time - order.created_at).total_seconds()
                    if age_seconds > self.order_timeout:
                        expired_orders.append(order_id)
                    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                        self.active_orders.pop(order_id, None)
                        self._notify_callbacks(order, 'completed')
                for order_id in expired_orders:
                    order = self.active_orders.get(order_id)
                    if order:
                        order.status = OrderStatus.EXPIRED
                        order.updated_at = current_time
                        self.active_orders.pop(order_id, None)
                        logger.warning(f'Order {order_id} expired after {self.order_timeout} seconds')
                        self._notify_callbacks(order, 'expired')
                from .reconcile import reconcile_positions_and_orders
                reconcile_positions_and_orders()
                time.sleep(1)
            except (APIError, TimeoutError, ConnectionError) as e:
                logger.error('ORDER_MONITOR_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
                time.sleep(5)

    def _notify_callbacks(self, order: Order, event_type: str):
        """Notify registered callbacks of order events."""
        try:
            for callback in self.execution_callbacks:
                try:
                    callback(order, event_type)
                except (KeyError, ValueError, TypeError, RuntimeError) as e:
                    logger.error('CALLBACK_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'order_id': order.id})
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.error('CALLBACK_NOTIFICATION_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'order_id': order.id})

class ExecutionEngine:
    """
    Main execution engine for institutional order processing.

    Coordinates order management, execution algorithms,
    and provides unified execution interface.
    """
    _minute_stats: dict[str, float] = {}
    _latest_quote: dict[str, float] = {}

    def __init__(self, market_data_feed=None, broker_interface=None):
        """Initialize execution engine."""
        self.order_manager = OrderManager()
        self.market_data_feed = market_data_feed
        self.broker_interface = broker_interface
        self.logger = logger
        self._open_orders: dict[str, OrderInfo] = {}
        self.execution_stats = {'total_orders': 0, 'filled_orders': 0, 'cancelled_orders': 0, 'rejected_orders': 0, 'total_volume': 0.0, 'average_fill_time': 0.0}
        emit_once(logger, 'EXECUTION_ENGINE_INIT', 'info', 'ExecutionEngine initialized')

    def _track_order(self, order: Order) -> None:
        """Track an order in the shared monitoring structure."""
        _cleanup_stale_orders()
        info = OrderInfo(
            order_id=order.id,
            symbol=order.symbol,
            side=getattr(order.side, 'value', order.side),
            qty=getattr(order, 'quantity', getattr(order, 'qty', 0)),
            submitted_time=time.time(),
            last_status=getattr(order.status, 'value', getattr(order, 'status', 'new')),
        )
        with _order_tracking_lock:
            _active_orders[order.id] = info

    def track_order(self, order: Order) -> None:
        """Public wrapper for :meth:`_track_order`."""
        self._track_order(order)

    def _update_order_status(self, order_id: str, status: str) -> None:
        """Update tracked order status and remove if terminal."""
        terminal = {'filled', 'canceled', 'cancelled', 'rejected'}
        with _order_tracking_lock:
            info = _active_orders.get(order_id)
            if info:
                info.last_status = status
                if status.lower() in terminal:
                    _active_orders.pop(order_id, None)

    def get_pending_orders(self) -> list[OrderInfo]:
        """Return list of currently tracked orders."""
        with _order_tracking_lock:
            return list(_active_orders.values())

    def _cancel_stale_order(self, order_id: str) -> bool:
        """Attempt to cancel a stale order via broker interface."""
        if self.broker_interface is None:
            return False
        try:
            ord_obj = self.broker_interface.get_order(order_id)
            if getattr(ord_obj, 'status', '').lower() == 'new':
                self.broker_interface.cancel_order(order_id)
            return True
        except Exception:
            return False

    def _assess_liquidity(self, symbol: str, quantity: int) -> tuple[int, bool]:
        """Assess liquidity and optionally adjust quantity."""
        bid, ask = (0.0, 0.0)
        try:
            bid, ask = self._latest_quote()
        except (RuntimeError, ValueError):
            return (quantity, False)
        spread_pct = (ask - bid) / bid if bid else 0.0
        if spread_pct >= 0.01:
            return (int(quantity * 0.75), False)
        return (quantity, False)

    def cleanup_stale_orders(self, now: float | None=None, max_age_seconds: int | None=None) -> int:
        """Remove stale orders and attempt cancelation via broker."""
        now_s = now if now is not None else time.time()
        max_age = max_age_seconds if max_age_seconds is not None else ORDER_STALE_AFTER_S
        with _order_tracking_lock:
            stale_ids = [oid for oid, info in _active_orders.items() if now_s - info.submitted_time >= max_age]
        for oid in stale_ids:
            self._cancel_stale_order(oid)
        return _cleanup_stale_orders(now_s, max_age)

    def check_stops(self) -> None:
        """
        Safety hook invoked after each cycle. It should never raise.
        For now: best-effort inspection of open positions; no-op if unsupported.
        """
        try:
            broker = getattr(self, 'broker', None) or getattr(self, 'broker_interface', None)
            if broker is None or not hasattr(broker, 'list_positions'):
                logger.debug('check_stops: no broker/list_positions; skipping')
                return
            positions = broker.list_positions() or []
            logger.debug('check_stops: inspected %d positions', len(positions))
        except (ValueError, TypeError) as e:
            logger.info('check_stops: suppressed exception: %s', e)

    def _validate_short_selling(self, symbol: str, qty: float, price: float) -> None:
        from ai_trading.risk.short_selling import validate_short_selling
        validate_short_selling(symbol, qty, price)

    def _reconcile_partial_fills(self, *args, requested_qty=None, remaining_qty=None, symbol=None, side=None, **_kwargs) -> None:
        """Detect partial fills and log for quantity tracking."""
        try:
            if requested_qty is None or remaining_qty is None:
                return
            filled = requested_qty - remaining_qty
            if filled < requested_qty:
                self.logger.info('PARTIAL_FILL_DETECTED', extra={'symbol': symbol, 'side': side, 'filled': filled, 'requested': requested_qty})
        except (ValueError, TypeError):
            pass

    def execute_order(self, symbol: str, quantity_or_side=None, side_or_quantity=None, order_type: OrderType=OrderType.MARKET, method=None, **kwargs):
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
            if isinstance(quantity_or_side, int):
                quantity = quantity_or_side
                side_str = side_or_quantity
                if isinstance(side_str, str):
                    if side_str.lower() == 'buy':
                        side = OrderSide.BUY
                    elif side_str.lower() == 'sell':
                        side = OrderSide.SELL
                    else:
                        raise ValueError(f'Invalid side: {side_str}')
                else:
                    side = side_str
            else:
                side = quantity_or_side
                quantity = side_or_quantity
            if method == 'twap':
                return self.execute_sliced(symbol, quantity, side, **kwargs)
            quantity = int(_ensure_positive_qty(quantity))
            kwargs['limit_price'] = _ensure_valid_price(kwargs.get('limit_price'))
            kwargs['stop_price'] = _ensure_valid_price(kwargs.get('stop_price'))
            payload: dict[str, Any] = {'symbol': symbol, 'side': getattr(side, 'value', side), 'qty': quantity, 'type': getattr(order_type, 'value', order_type), 'time_in_force': kwargs.get('time_in_force'), 'limit_price': kwargs.get('limit_price'), 'stop_price': kwargs.get('stop_price')}
            logger.debug('ORDER_SUBMIT_PAYLOAD', extra={k: payload.get(k) for k in ('symbol', 'side', 'qty', 'type', 'time_in_force', 'limit_price', 'stop_price')})
            order = Order(symbol, side, quantity, order_type, **kwargs)
            if self.order_manager.submit_order(order):
                self.execution_stats['total_orders'] += 1
                if order_type == OrderType.MARKET:
                    self._simulate_market_execution(order)
                return order.id
            else:
                self.execution_stats['rejected_orders'] += 1
                return None
        except (ValueError, TypeError, KeyError) as e:
            logger.error('EXECUTE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            self.execution_stats['rejected_orders'] += 1
            raise

    def execute_sliced(self, symbol: str, quantity: int, side: OrderSide, **kwargs):
        """Execute order slices by delegating to execute_order."""
        return self.execute_order(symbol, side, quantity, **kwargs)

    def _simulate_market_execution(self, order: Order):
        """Simulate market order execution (demo purposes)."""
        try:
            base_price = order.price or 100.0
            remaining = order.quantity
            while remaining > 0 and order.status != OrderStatus.CANCELED:
                fill_quantity = min(remaining, max(1, remaining // 3))
                fill_price = base_price * (1 + (hash(order.id) % 100 - 50) / 10000)
                order.add_fill(fill_quantity, fill_price)
                remaining -= fill_quantity
                if remaining > 0:
                    time.sleep(0.1)
            if order.is_filled:
                self.execution_stats['filled_orders'] += 1
                self.execution_stats['total_volume'] += order.notional_value
                fill_time = (order.executed_at - order.created_at).total_seconds()
                self.execution_stats['average_fill_time'] = (self.execution_stats['average_fill_time'] * (self.execution_stats['filled_orders'] - 1) + fill_time) / self.execution_stats['filled_orders']
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            logger.error('SIMULATION_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'order_id': order.id})

    def get_execution_stats(self) -> dict:
        """Get execution engine statistics."""
        stats = self.execution_stats.copy()
        stats['active_orders'] = len(self.order_manager.active_orders)
        stats['success_rate'] = stats['filled_orders'] / stats['total_orders'] if stats['total_orders'] > 0 else 0
        return stats
