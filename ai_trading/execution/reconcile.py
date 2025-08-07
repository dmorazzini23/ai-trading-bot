"""
Position and order reconciliation module.

Reconciles local trading state with broker truth by:
1. Pulling current broker positions and open orders
2. Comparing with local state 
3. Fixing drifts (cancel stale locals, resync quantities)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from ai_trading.core.interfaces import Position, Order, OrderSide, OrderStatus


logger = logging.getLogger(__name__)


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
    local_status: Optional[OrderStatus]
    broker_status: Optional[OrderStatus] 
    action_needed: str  # 'cancel_local', 'update_local', 'none'


@dataclass
class ReconciliationResult:
    """Result of position/order reconciliation."""
    position_drifts: List[PositionDrift]
    order_drifts: List[OrderDrift]
    actions_taken: List[str]
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
    
    def __init__(self, tolerance_pct: float = 0.01, min_drift_qty: float = 0.001):
        """
        Initialize reconciler.
        
        Args:
            tolerance_pct: Position drift tolerance as percentage
            min_drift_qty: Minimum quantity drift to trigger reconciliation
        """
        self.tolerance_pct = tolerance_pct
        self.min_drift_qty = min_drift_qty
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def reconcile_positions(
        self,
        local_positions: Dict[str, Position],
        broker_positions: Dict[str, Position]
    ) -> List[PositionDrift]:
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
            
            # Calculate drift percentage
            base_qty = max(abs(local_qty), abs(broker_qty), 1.0)  # Avoid div by zero
            drift_pct = drift_qty / base_qty
            
            # Check if drift is significant
            if abs(drift_pct) > self.tolerance_pct or abs(drift_qty) > self.min_drift_qty:
                drift = PositionDrift(
                    symbol=symbol,
                    local_qty=local_qty,
                    broker_qty=broker_qty,
                    drift_qty=drift_qty,
                    drift_pct=drift_pct
                )
                drifts.append(drift)
                
                self.logger.warning(
                    f"Position drift detected for {symbol}: "
                    f"local={local_qty}, broker={broker_qty}, "
                    f"drift={drift_qty} ({drift_pct:.2%})"
                )
        
        return drifts
    
    def reconcile_orders(
        self,
        local_orders: Dict[str, Order],
        broker_orders: Dict[str, Order]
    ) -> List[OrderDrift]:
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
            
            # Determine action needed
            action_needed = "none"
            
            if local_order and not broker_order:
                # Local order doesn't exist at broker - cancel local
                action_needed = "cancel_local"
            elif not local_order and broker_order:
                # Broker order not tracked locally - add to local
                action_needed = "add_local"
            elif local_status != broker_status:
                # Status mismatch - update local to match broker
                action_needed = "update_local"
            
            if action_needed != "none":
                symbol = (local_order.symbol if local_order 
                         else broker_order.symbol if broker_order else "UNKNOWN")
                
                drift = OrderDrift(
                    order_id=order_id,
                    symbol=symbol,
                    local_status=local_status,
                    broker_status=broker_status,
                    action_needed=action_needed
                )
                drifts.append(drift)
                
                self.logger.warning(
                    f"Order drift detected for {order_id} ({symbol}): "
                    f"local_status={local_status}, broker_status={broker_status}, "
                    f"action={action_needed}"
                )
        
        return drifts
    
    def apply_position_fixes(
        self, 
        drifts: List[PositionDrift],
        position_manager
    ) -> List[str]:
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
                # Update local position to match broker
                position_manager.sync_position(drift.symbol, drift.broker_qty)
                
                action = (f"Synced position for {drift.symbol}: "
                         f"{drift.local_qty} -> {drift.broker_qty}")
                actions.append(action)
                self.logger.info(action)
                
            except Exception as e:
                error_action = f"Failed to sync position for {drift.symbol}: {e}"
                actions.append(error_action)
                self.logger.error(error_action)
        
        return actions
    
    def apply_order_fixes(
        self,
        drifts: List[OrderDrift], 
        order_manager
    ) -> List[str]:
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
                if drift.action_needed == "cancel_local":
                    # Remove stale local order
                    order_manager.remove_local_order(drift.order_id)
                    action = f"Removed stale local order {drift.order_id}"
                    
                elif drift.action_needed == "update_local":
                    # Update local order status to match broker
                    order_manager.update_order_status(drift.order_id, drift.broker_status)
                    action = f"Updated order {drift.order_id} status to {drift.broker_status}"
                    
                elif drift.action_needed == "add_local":
                    # Add broker order to local tracking
                    # Note: This would need broker order details
                    action = f"Need to add broker order {drift.order_id} to local tracking"
                    
                else:
                    continue
                    
                actions.append(action)
                self.logger.info(action)
                
            except Exception as e:
                error_action = f"Failed to fix order {drift.order_id}: {e}"
                actions.append(error_action)
                self.logger.error(error_action)
        
        return actions
    
    def full_reconciliation(
        self,
        local_positions: Dict[str, Position],
        broker_positions: Dict[str, Position], 
        local_orders: Dict[str, Order],
        broker_orders: Dict[str, Order],
        position_manager=None,
        order_manager=None,
        apply_fixes: bool = True
    ) -> ReconciliationResult:
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
        self.logger.info("Starting full reconciliation")
        
        # Find drifts
        position_drifts = self.reconcile_positions(local_positions, broker_positions)
        order_drifts = self.reconcile_orders(local_orders, broker_orders)
        
        actions = []
        
        # Apply fixes if requested and managers provided
        if apply_fixes:
            if position_manager and position_drifts:
                actions.extend(self.apply_position_fixes(position_drifts, position_manager))
                
            if order_manager and order_drifts:
                actions.extend(self.apply_order_fixes(order_drifts, order_manager))
        
        result = ReconciliationResult(
            position_drifts=position_drifts,
            order_drifts=order_drifts,
            actions_taken=actions,
            reconciled_at=datetime.now(timezone.utc)
        )
        
        self.logger.info(
            f"Reconciliation complete: {len(position_drifts)} position drifts, "
            f"{len(order_drifts)} order drifts, {len(actions)} actions taken"
        )
        
        return result


# Global reconciler instance
_global_reconciler: Optional[PositionReconciler] = None


def get_reconciler() -> PositionReconciler:
    """Get or create global reconciler instance."""
    global _global_reconciler
    if _global_reconciler is None:
        _global_reconciler = PositionReconciler()
    return _global_reconciler


def reconcile_with_broker(
    broker_client,
    local_positions: Dict[str, Position],
    local_orders: Dict[str, Order],
    apply_fixes: bool = True
) -> ReconciliationResult:
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
    # This would need to be implemented based on specific broker client interface
    # For now, return empty result
    reconciler = get_reconciler()
    
    # Placeholder - would need actual broker client integration
    broker_positions = {}  # broker_client.get_positions()
    broker_orders = {}     # broker_client.get_open_orders()
    
    return reconciler.full_reconciliation(
        local_positions=local_positions,
        broker_positions=broker_positions,
        local_orders=local_orders,
        broker_orders=broker_orders,
        apply_fixes=apply_fixes
    )


def reconcile_positions_and_orders() -> ReconciliationResult:
    """
    Convenience function to run reconciliation with current state.
    
    This function is called from the execution engine to reconcile
    positions and orders after trading activity.
    
    Returns:
        ReconciliationResult with any detected drifts
    """
    try:
        # In a real implementation, this would:
        # 1. Get current local positions and orders from the execution engine
        # 2. Fetch current broker positions and orders
        # 3. Run reconciliation and apply fixes
        
        # For now, return empty result to avoid errors
        logger.debug("Position/order reconciliation called (mock implementation)")
        return ReconciliationResult(
            position_drifts=[],
            order_drifts=[],
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Error in reconciliation: {e}")
        return ReconciliationResult(
            position_drifts=[],
            order_drifts=[], 
            timestamp=datetime.now(timezone.utc)
        )