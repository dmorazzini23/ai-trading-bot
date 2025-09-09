"""
Position and order reconciliation module.

Reconciles local trading state with broker truth by:
1. Pulling current broker positions and open orders
2. Comparing with local state
3. Fixing drifts (cancel stale locals, resync quantities)
"""
from ai_trading.logging import get_logger
from dataclasses import dataclass
from datetime import UTC, datetime
from ai_trading.core.interfaces import Order, OrderStatus, Position, OrderType
from ai_trading.order.types import OrderSide
logger = get_logger(__name__)

@dataclass
class PositionDrift:
    """Represents a drift between local and broker position."""
    symbol: str
    local_qty: float
    broker_qty: float
    drift_qty: float
    drift_pct: float

    @property
    def is_significant(self) -> bool:
        """Check if drift is significant enough to reconcile."""
        return abs(self.drift_pct) > 0.01 or abs(self.drift_qty) > 0.001

@dataclass
class OrderDrift:
    """Represents a drift between local and broker orders."""
    order_id: str
    symbol: str
    local_status: OrderStatus | None
    broker_status: OrderStatus | None
    action_needed: str

@dataclass
class ReconciliationResult:
    """Result of position/order reconciliation."""
    position_drifts: list[PositionDrift]
    order_drifts: list[OrderDrift]
    actions_taken: list[str]
    reconciled_at: datetime

    @property
    def has_drifts(self) -> bool:
        """Check if any drifts were found."""
        return len(self.position_drifts) > 0 or len(self.order_drifts) > 0

class PositionReconciler:
    """
    Reconciles trading positions and orders with broker truth.

    Handles:
    - Position quantity mismatches
    - Stale local orders that don't exist at broker
    - Order status synchronization
    """

    def __init__(self, tolerance_pct: float=0.01, min_drift_qty: float=0.001):
        """
        Initialize reconciler.

        Args:
            tolerance_pct: Position drift tolerance as percentage
            min_drift_qty: Minimum quantity drift to trigger reconciliation
        """
        self.tolerance_pct = tolerance_pct
        self.min_drift_qty = min_drift_qty
        self.logger = get_logger(f'{__name__}.{self.__class__.__name__}')

    def reconcile_positions(self, local_positions: dict[str, Position], broker_positions: dict[str, Position]) -> list[PositionDrift]:
        """
        Compare local vs broker positions and identify drifts.

        Args:
            local_positions: Local position tracking {symbol: Position}
            broker_positions: Broker positions {symbol: Position}

        Returns:
            List of position drifts found
        """
        drifts = []
        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
        for symbol in all_symbols:
            local_pos = local_positions.get(symbol)
            broker_pos = broker_positions.get(symbol)
            local_qty = local_pos.quantity if local_pos else 0.0
            broker_qty = broker_pos.quantity if broker_pos else 0.0
            drift_qty = local_qty - broker_qty
            base_qty = max(abs(local_qty), abs(broker_qty), 1.0)
            drift_pct = drift_qty / base_qty
            if abs(drift_pct) > self.tolerance_pct or abs(drift_qty) > self.min_drift_qty:
                drift = PositionDrift(symbol=symbol, local_qty=local_qty, broker_qty=broker_qty, drift_qty=drift_qty, drift_pct=drift_pct)
                drifts.append(drift)
                self.logger.warning(f'Position drift detected for {symbol}: local={local_qty}, broker={broker_qty}, drift={drift_qty} ({drift_pct:.2%})')
        return drifts

    def reconcile_orders(self, local_orders: dict[str, Order], broker_orders: dict[str, Order]) -> list[OrderDrift]:
        """
        Compare local vs broker orders and identify drifts.

        Args:
            local_orders: Local order tracking {order_id: Order}
            broker_orders: Broker orders {order_id: Order}

        Returns:
            List of order drifts found
        """
        drifts = []
        all_order_ids = set(local_orders.keys()) | set(broker_orders.keys())
        for order_id in all_order_ids:
            local_order = local_orders.get(order_id)
            broker_order = broker_orders.get(order_id)
            local_status = local_order.status if local_order else None
            broker_status = broker_order.status if broker_order else None
            action_needed = 'none'
            if local_order and (not broker_order):
                action_needed = 'cancel_local'
            elif not local_order and broker_order:
                action_needed = 'add_local'
            elif local_status != broker_status:
                action_needed = 'update_local'
            if action_needed != 'none':
                symbol = local_order.symbol if local_order else broker_order.symbol if broker_order else 'UNKNOWN'
                drift = OrderDrift(order_id=order_id, symbol=symbol, local_status=local_status, broker_status=broker_status, action_needed=action_needed)
                drifts.append(drift)
                self.logger.warning(f'Order drift detected for {order_id} ({symbol}): local_status={local_status}, broker_status={broker_status}, action={action_needed}')
        return drifts

    def apply_position_fixes(self, drifts: list[PositionDrift], position_manager) -> list[str]:
        """
        Apply fixes for position drifts.

        Args:
            drifts: Position drifts to fix
            position_manager: Position manager to update

        Returns:
            List of actions taken
        """
        actions = []
        for drift in drifts:
            if not drift.is_significant:
                continue
            try:
                position_manager.sync_position(drift.symbol, drift.broker_qty)
                action = f'Synced position for {drift.symbol}: {drift.local_qty} -> {drift.broker_qty}'
                actions.append(action)
                self.logger.info(action)
            except (RuntimeError, ValueError) as e:
                error_action = f'Failed to sync position for {drift.symbol}: {e}'
                actions.append(error_action)
                self.logger.error(error_action)
        return actions

    def apply_order_fixes(self, drifts: list[OrderDrift], order_manager) -> list[str]:
        """
        Apply fixes for order drifts.

        Args:
            drifts: Order drifts to fix
            order_manager: Order manager to update

        Returns:
            List of actions taken
        """
        actions = []
        for drift in drifts:
            try:
                if drift.action_needed == 'cancel_local':
                    order_manager.remove_local_order(drift.order_id)
                    action = f'Removed stale local order {drift.order_id}'
                elif drift.action_needed == 'update_local':
                    order_manager.update_order_status(drift.order_id, drift.broker_status)
                    action = f'Updated order {drift.order_id} status to {drift.broker_status}'
                elif drift.action_needed == 'add_local':
                    action = f'Need to add broker order {drift.order_id} to local tracking'
                else:
                    continue
                actions.append(action)
                self.logger.info(action)
            except (RuntimeError, ValueError) as e:
                error_action = f'Failed to fix order {drift.order_id}: {e}'
                actions.append(error_action)
                self.logger.error(error_action)
        return actions

    def full_reconciliation(self, local_positions: dict[str, Position], broker_positions: dict[str, Position], local_orders: dict[str, Order], broker_orders: dict[str, Order], position_manager=None, order_manager=None, apply_fixes: bool=True) -> ReconciliationResult:
        """
        Perform full position and order reconciliation.

        Args:
            local_positions: Local position state
            broker_positions: Broker position state
            local_orders: Local order state
            broker_orders: Broker order state
            position_manager: Manager to apply position fixes
            order_manager: Manager to apply order fixes
            apply_fixes: Whether to automatically apply fixes

        Returns:
            ReconciliationResult with drifts and actions taken
        """
        self.logger.info('Starting full reconciliation')
        position_drifts = self.reconcile_positions(local_positions, broker_positions)
        order_drifts = self.reconcile_orders(local_orders, broker_orders)
        actions = []
        if apply_fixes:
            if position_manager and position_drifts:
                actions.extend(self.apply_position_fixes(position_drifts, position_manager))
            if order_manager and order_drifts:
                actions.extend(self.apply_order_fixes(order_drifts, order_manager))
        result = ReconciliationResult(position_drifts=position_drifts, order_drifts=order_drifts, actions_taken=actions, reconciled_at=datetime.now(UTC))
        self.logger.info(f'Reconciliation complete: {len(position_drifts)} position drifts, {len(order_drifts)} order drifts, {len(actions)} actions taken')
        return result
_global_reconciler: PositionReconciler | None = None

def get_reconciler() -> PositionReconciler:
    """Get or create global reconciler instance."""
    global _global_reconciler
    if _global_reconciler is None:
        _global_reconciler = PositionReconciler()
    return _global_reconciler

def reconcile_with_broker(broker_client, local_positions: dict[str, Position], local_orders: dict[str, Order], apply_fixes: bool=True) -> ReconciliationResult:
    """
    Convenience function to reconcile with broker using client.

    Args:
        broker_client: Broker client to fetch positions/orders
        local_positions: Local position state
        local_orders: Local order state
        apply_fixes: Whether to apply fixes automatically

    Returns:
        ReconciliationResult
    """
    reconciler = get_reconciler()
    broker_positions: dict[str, Position] = {}
    broker_orders: dict[str, Order] = {}

    # Fetch current broker positions
    try:
        positions = broker_client.list_positions() or []
        for pos in positions:
            qty = int(getattr(pos, "qty", getattr(pos, "quantity", 0)))
            broker_positions[pos.symbol] = Position(
                symbol=pos.symbol,
                quantity=qty,
                market_value=float(getattr(pos, "market_value", 0.0)),
                cost_basis=float(getattr(pos, "cost_basis", 0.0)),
                unrealized_pnl=float(getattr(pos, "unrealized_pl", 0.0)),
                timestamp=datetime.now(UTC),
            )
    except Exception as e:  # pragma: no cover - network issues
        logger.error(f"Failed to fetch broker positions: {e}")

    # Fetch open broker orders if supported
    try:
        if hasattr(broker_client, "list_orders"):
            orders = broker_client.list_orders(status="open") or []
        else:
            orders = []
        for ord_obj in orders:
            status = OrderStatus(getattr(ord_obj, "status"))
            qty = int(getattr(ord_obj, "qty", getattr(ord_obj, "quantity", 0)))
            filled_qty = int(getattr(ord_obj, "filled_qty", getattr(ord_obj, "filled_quantity", 0)))
            broker_orders[ord_obj.id] = Order(
                id=ord_obj.id,
                symbol=getattr(ord_obj, "symbol", ""),
                side=getattr(ord_obj, "side", OrderSide.BUY),
                order_type=getattr(ord_obj, "order_type", OrderType.MARKET),
                status=status,
                quantity=qty,
                filled_quantity=filled_qty,
                price=getattr(ord_obj, "limit_price", getattr(ord_obj, "price", None)),
                filled_price=getattr(ord_obj, "filled_avg_price", None),
                timestamp=datetime.now(UTC),
            )
    except Exception as e:  # pragma: no cover - network issues
        logger.error(f"Failed to fetch broker orders: {e}")

    return reconciler.full_reconciliation(
        local_positions=local_positions,
        broker_positions=broker_positions,
        local_orders=local_orders,
        broker_orders=broker_orders,
        apply_fixes=apply_fixes,
    )

def reconcile_positions_and_orders(ctx=None) -> ReconciliationResult:
    """Synchronize local state with broker truth.

    Parameters
    ----------
    ctx:
        Optional execution context providing ``api`` (broker client),
        ``positions`` (``dict[str, float]``) and ``orders`` (``list[Order]``).
        When ``None`` no reconciliation is performed and an empty result is
        returned.  This keeps the function safe for callers that do not yet
        provide a context.

    Returns
    -------
    ReconciliationResult
        Reconciliation outcome including any detected drifts.  The result's
        ``reconciled_at`` timestamp reflects when the reconciliation occurred.
    """
    broker_client = getattr(ctx, "api", None) if ctx else None
    if broker_client is None:
        logger.debug("No broker context available for reconciliation")
        return ReconciliationResult([], [], [], datetime.now(UTC))

    # Extract local state
    local_positions: dict[str, int] = {}
    for sym, pos in (getattr(ctx, "positions", {}) or {}).items():
        qty = pos.quantity if isinstance(pos, Position) else int(pos)
        local_positions[sym] = qty

    local_orders_list = list(getattr(ctx, "orders", []) or [])
    local_orders: dict[str, Order] = {o.id: o for o in local_orders_list}

    # Update local orders with broker fill information
    for order in local_orders.values():
        try:
            broker_order = broker_client.get_order(order.id)
            broker_status = OrderStatus(getattr(broker_order, "status"))
            filled_qty = int(
                getattr(broker_order, "filled_qty", getattr(broker_order, "filled_quantity", order.filled_quantity))
            )
            if order.status != broker_status or order.filled_quantity != filled_qty:
                order.status = broker_status
                order.filled_quantity = filled_qty
                order.timestamp = datetime.now(UTC)
                if broker_status == OrderStatus.FILLED:
                    side_mult = 1 if getattr(order.side, "value", order.side) == OrderSide.BUY.value else -1
                    local_positions[order.symbol] = local_positions.get(order.symbol, 0) + side_mult * filled_qty
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to update order {order.id}: {e}")

    # Perform reconciliation against broker state
    result = reconcile_with_broker(
        broker_client,
        local_positions={
            sym: Position(
                symbol=sym,
                quantity=qty,
                market_value=0.0,
                cost_basis=0.0,
                unrealized_pnl=0.0,
                timestamp=datetime.now(UTC),
            )
            for sym, qty in local_positions.items()
        },
        local_orders=local_orders,
        apply_fixes=False,
    )

    # Apply position drift fixes locally
    for drift in result.position_drifts:
        local_positions[drift.symbol] = drift.broker_qty

    # Persist updated state back to context
    ctx.positions = local_positions
    ctx.orders = list(local_orders.values())
    return result
