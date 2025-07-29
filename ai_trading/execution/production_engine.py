"""
Production execution coordinator that integrates risk management and monitoring.

Enhances the existing execution engine with comprehensive safety checks,
real-time monitoring, and advanced risk management capabilities.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .engine import Order, OrderStatus, ExecutionAlgorithm
from ..core.enums import OrderSide, OrderType, RiskLevel
from ..core.constants import EXECUTION_PARAMETERS, RISK_PARAMETERS
from ..risk import (
    DynamicPositionSizer,
    TradingHaltManager,
    RiskManager
)
from ..monitoring import AlertManager, AlertSeverity


class ProductionExecutionCoordinator:
    """
    Production execution coordinator with comprehensive safety integration.
    
    Wraps the core execution engine with advanced risk management,
    monitoring, and safety mechanisms for production trading.
    """
    
    def __init__(self, account_equity: float, risk_level: RiskLevel = RiskLevel.MODERATE):
        """Initialize production execution coordinator."""
        # AI-AGENT-REF: Production execution coordinator with comprehensive safety
        self.account_equity = account_equity
        self.risk_level = risk_level
        
        # Risk management components
        self.position_sizer = DynamicPositionSizer(risk_level)
        self.halt_manager = TradingHaltManager()
        self.risk_manager = RiskManager(risk_level)
        
        # Monitoring and alerting
        self.alert_manager = AlertManager()
        
        # Execution tracking
        self.pending_orders = {}
        self.completed_orders = {}
        self.rejected_orders = {}
        
        # Performance metrics
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "rejected_orders": 0,
            "average_execution_time_ms": 0.0,
            "total_slippage_bps": 0.0,
            "last_reset": datetime.now()
        }
        
        # Position tracking
        self.current_positions = {}
        
        logger.info(f"ProductionExecutionCoordinator initialized with equity=${account_equity:,.2f}")
    
    async def submit_order(self, symbol: str, side: OrderSide, quantity: int,
                          order_type: OrderType = OrderType.MARKET,
                          price: Optional[float] = None,
                          strategy: str = "unknown",
                          metadata: Dict = None) -> Dict[str, Any]:
        """
        Submit order with comprehensive safety checks and optimization.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type
            price: Limit price (if applicable)
            strategy: Strategy name for tracking
            metadata: Additional order metadata
            
        Returns:
            Order execution result
        """
        try:
            start_time = time.time()
            
            # Step 1: Create order object
            order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                strategy_id=strategy,
                **metadata or {}
            )
            
            # Step 2: Pre-execution safety checks
            safety_result = await self._comprehensive_safety_check(order)
            if not safety_result["approved"]:
                await self._handle_order_rejection(order, safety_result["reason"])
                return self._create_order_result(order, "rejected", safety_result["reason"])
            
            # Step 3: Position sizing optimization
            sizing_result = await self._optimize_order_size(order)
            if sizing_result["final_quantity"] == 0:
                await self._handle_order_rejection(order, "Position sizing resulted in zero quantity")
                return self._create_order_result(order, "rejected", "Invalid position size")
            
            # Update order with optimized quantity
            original_quantity = order.quantity
            order.quantity = sizing_result["final_quantity"]
            
            # Step 4: Market impact analysis
            impact_analysis = await self._analyze_market_impact(order)
            
            # Step 5: Execute order
            execution_result = await self._execute_order_with_monitoring(order, impact_analysis)
            
            # Step 6: Post-execution processing
            await self._post_execution_processing(order, execution_result, original_quantity)
            
            # Update statistics
            execution_time_ms = (time.time() - start_time) * 1000
            await self._update_execution_statistics(execution_result, execution_time_ms)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in order submission: {e}")
            return {"status": "error", "message": f"Submission error: {e}"}
    
    async def _comprehensive_safety_check(self, order: Order) -> Dict[str, Any]:
        """Perform comprehensive safety checks before execution."""
        try:
            # Check trading halt status
            halt_status = self.halt_manager.is_trading_allowed()
            if not halt_status["trading_allowed"]:
                return {
                    "approved": False,
                    "reason": f"Trading halted: {', '.join(halt_status['reasons'])}"
                }
            
            # Basic order validation
            if order.quantity <= 0:
                return {"approved": False, "reason": "Invalid quantity"}
            
            if order.quantity > EXECUTION_PARAMETERS["MAX_ORDER_SIZE"]:
                return {"approved": False, "reason": "Quantity exceeds maximum limit"}
            
            # Risk assessment
            position_history = self._get_symbol_history(order.symbol)
            risk_assessment = self.risk_manager.assess_trade_risk(
                order.symbol,
                order.quantity,
                order.price or 100.0,  # Use default price if not provided
                self.account_equity,
                position_history
            )
            
            if not risk_assessment["approved"]:
                return {
                    "approved": False,
                    "reason": f"Risk assessment failed: {', '.join(risk_assessment['warnings'])}"
                }
            
            # Check for recent similar orders (prevent duplicate submissions)
            if self._has_recent_similar_order(order):
                return {"approved": False, "reason": "Similar order recently submitted"}
            
            return {"approved": True, "reason": "All safety checks passed"}
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return {"approved": False, "reason": f"Safety check error: {e}"}
    
    async def _optimize_order_size(self, order: Order) -> Dict[str, Any]:
        """Optimize order size using dynamic position sizing."""
        try:
            # Get market data (simplified for this implementation)
            market_data = {
                "current_price": order.price or 100.0,
                "atr": 2.0,  # Default ATR
                "volume": 1000000  # Default volume
            }
            
            # Get historical data
            historical_data = {
                "returns": [],
                "trade_history": []
            }
            
            # Calculate optimal position size
            sizing_result = self.position_sizer.calculate_optimal_position(
                order.symbol,
                self.account_equity,
                market_data["current_price"],
                market_data,
                historical_data
            )
            
            # Apply any halt manager position size multipliers
            halt_status = self.halt_manager.is_trading_allowed()
            position_multiplier = halt_status.get("position_size_multiplier", 1.0)
            
            # Calculate final quantity
            final_quantity = min(
                order.quantity,
                sizing_result["recommended_size"],
                int(order.quantity * position_multiplier)
            )
            
            return {
                "original_quantity": order.quantity,
                "recommended_quantity": sizing_result["recommended_size"],
                "final_quantity": max(0, final_quantity),
                "position_multiplier": position_multiplier,
                "sizing_warnings": sizing_result.get("warnings", [])
            }
            
        except Exception as e:
            logger.error(f"Error optimizing order size: {e}")
            return {"final_quantity": 0}
    
    async def _analyze_market_impact(self, order: Order) -> Dict[str, Any]:
        """Analyze potential market impact of the order."""
        try:
            # Simplified market impact analysis
            current_price = order.price or 100.0
            notional_value = order.quantity * current_price
            
            # Estimate impact based on order size
            if notional_value > 1000000:  # $1M+
                impact_level = "high"
                estimated_slippage_bps = 15
            elif notional_value > 100000:  # $100K+
                impact_level = "medium"
                estimated_slippage_bps = 8
            else:
                impact_level = "low"
                estimated_slippage_bps = 3
            
            return {
                "impact_level": impact_level,
                "estimated_slippage_bps": estimated_slippage_bps,
                "notional_value": notional_value,
                "recommended_algorithm": self._recommend_execution_algorithm(impact_level)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market impact: {e}")
            return {"impact_level": "unknown", "estimated_slippage_bps": 0}
    
    async def _execute_order_with_monitoring(self, order: Order, 
                                           impact_analysis: Dict) -> Dict[str, Any]:
        """Execute order with real-time monitoring."""
        try:
            # Add to pending orders
            self.pending_orders[order.id] = order
            
            # Simulate order execution (in production, integrate with actual broker API)
            await asyncio.sleep(0.1)  # Simulate execution delay
            
            # Simulate fill
            fill_price = order.price or 100.0
            if order.side == OrderSide.BUY:
                fill_price *= (1 + impact_analysis["estimated_slippage_bps"] / 10000)
            else:
                fill_price *= (1 - impact_analysis["estimated_slippage_bps"] / 10000)
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.executed_at = datetime.now()
            
            # Move to completed orders
            self.completed_orders[order.id] = order
            del self.pending_orders[order.id]
            
            # Update position tracking
            self._update_position_tracking(order)
            
            # Calculate actual slippage
            expected_price = order.price or fill_price
            actual_slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
            
            return {
                "status": "success",
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": order.quantity,
                "fill_price": fill_price,
                "actual_slippage_bps": actual_slippage_bps,
                "execution_time": order.executed_at,
                "message": f"Order executed successfully at ${fill_price:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            self.rejected_orders[order.id] = order
            if order.id in self.pending_orders:
                del self.pending_orders[order.id]
            
            return {
                "status": "failed",
                "order_id": order.id,
                "message": f"Execution failed: {e}"
            }
    
    async def _post_execution_processing(self, order: Order, execution_result: Dict,
                                       original_quantity: int):
        """Handle post-execution processing and notifications."""
        try:
            if execution_result["status"] == "success":
                # Record trade with halt manager
                self.halt_manager.record_trade()
                
                # Send alerts for significant orders
                notional_value = order.quantity * execution_result["fill_price"]
                if notional_value > 50000:  # Alert for orders > $50k
                    await self.alert_manager.send_trading_alert(
                        "Order Executed",
                        order.symbol,
                        {
                            "side": order.side.value,
                            "quantity": order.quantity,
                            "fill_price": execution_result["fill_price"],
                            "notional_value": notional_value,
                            "slippage_bps": execution_result.get("actual_slippage_bps", 0)
                        },
                        AlertSeverity.INFO
                    )
                
                # Check for excessive slippage
                slippage = execution_result.get("actual_slippage_bps", 0)
                if slippage > EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"]:
                    await self.alert_manager.send_performance_alert(
                        "Excessive Slippage",
                        slippage,
                        EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"],
                        AlertSeverity.WARNING
                    )
                
                logger.info(f"Order {order.id} executed successfully: {order.symbol} "
                           f"{order.side.value} {order.quantity} @ ${execution_result['fill_price']:.2f}")
            
            else:
                # Handle execution failure
                await self.alert_manager.send_trading_alert(
                    "Order Execution Failed",
                    order.symbol,
                    {
                        "order_id": order.id,
                        "reason": execution_result.get("message", "Unknown"),
                        "quantity": order.quantity
                    },
                    AlertSeverity.WARNING
                )
                
                logger.warning(f"Order {order.id} execution failed: {execution_result.get('message')}")
            
        except Exception as e:
            logger.error(f"Error in post-execution processing: {e}")
    
    async def _handle_order_rejection(self, order: Order, reason: str):
        """Handle order rejection with proper logging and alerts."""
        try:
            order.status = OrderStatus.REJECTED
            self.rejected_orders[order.id] = order
            
            # Send rejection alert
            await self.alert_manager.send_trading_alert(
                "Order Rejected",
                order.symbol,
                {
                    "order_id": order.id,
                    "reason": reason,
                    "quantity": order.quantity
                },
                AlertSeverity.WARNING
            )
            
            logger.warning(f"Order {order.id} rejected: {reason}")
            
        except Exception as e:
            logger.error(f"Error handling order rejection: {e}")
    
    def _create_order_result(self, order: Order, status: str, message: str) -> Dict[str, Any]:
        """Create standardized order result."""
        return {
            "status": status,
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "message": message,
            "timestamp": datetime.now()
        }
    
    def _update_position_tracking(self, order: Order):
        """Update internal position tracking."""
        try:
            symbol = order.symbol
            quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
            fill_price = order.average_fill_price
            
            if symbol in self.current_positions:
                current_pos = self.current_positions[symbol]
                new_quantity = current_pos["quantity"] + quantity
                
                if new_quantity == 0:
                    del self.current_positions[symbol]
                else:
                    # Calculate new average price
                    total_cost = (current_pos["quantity"] * current_pos["avg_price"] + 
                                quantity * fill_price)
                    new_avg_price = total_cost / new_quantity
                    
                    self.current_positions[symbol] = {
                        "quantity": new_quantity,
                        "avg_price": new_avg_price,
                        "last_updated": datetime.now()
                    }
            else:
                if quantity != 0:
                    self.current_positions[symbol] = {
                        "quantity": quantity,
                        "avg_price": fill_price,
                        "last_updated": datetime.now()
                    }
                    
        except Exception as e:
            logger.error(f"Error updating position tracking: {e}")
    
    async def _update_execution_statistics(self, execution_result: Dict, execution_time_ms: float):
        """Update execution performance statistics."""
        try:
            self.execution_stats["total_orders"] += 1
            
            if execution_result["status"] == "success":
                self.execution_stats["successful_orders"] += 1
                
                # Update average execution time
                alpha = 0.1
                if self.execution_stats["average_execution_time_ms"] == 0:
                    self.execution_stats["average_execution_time_ms"] = execution_time_ms
                else:
                    self.execution_stats["average_execution_time_ms"] = (
                        alpha * execution_time_ms + 
                        (1 - alpha) * self.execution_stats["average_execution_time_ms"]
                    )
                
                # Track slippage
                slippage = execution_result.get("actual_slippage_bps", 0)
                self.execution_stats["total_slippage_bps"] += slippage
                
            else:
                self.execution_stats["rejected_orders"] += 1
                
        except Exception as e:
            logger.error(f"Error updating execution statistics: {e}")
    
    def _recommend_execution_algorithm(self, impact_level: str) -> ExecutionAlgorithm:
        """Recommend execution algorithm based on impact analysis."""
        if impact_level == "high":
            return ExecutionAlgorithm.TWAP  # Time-weighted for large orders
        elif impact_level == "medium":
            return ExecutionAlgorithm.VWAP  # Volume-weighted for medium orders
        else:
            return ExecutionAlgorithm.MARKET  # Market orders for small orders
    
    def _has_recent_similar_order(self, order: Order) -> bool:
        """Check for recent similar orders to prevent duplicates."""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=30)
            
            for existing_order in self.pending_orders.values():
                if (existing_order.symbol == order.symbol and
                    existing_order.side == order.side and
                    existing_order.quantity == order.quantity and
                    existing_order.created_at >= cutoff_time):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking for similar orders: {e}")
            return False
    
    def _get_symbol_history(self, symbol: str) -> List[Dict]:
        """Get trading history for a symbol."""
        # In production, this would query historical data
        return []
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution engine summary statistics."""
        try:
            success_rate = (
                self.execution_stats["successful_orders"] / self.execution_stats["total_orders"] * 100
                if self.execution_stats["total_orders"] > 0 else 0
            )
            
            avg_slippage = (
                self.execution_stats["total_slippage_bps"] / self.execution_stats["successful_orders"]
                if self.execution_stats["successful_orders"] > 0 else 0
            )
            
            return {
                "execution_stats": self.execution_stats,
                "success_rate_pct": success_rate,
                "average_slippage_bps": avg_slippage,
                "pending_orders": len(self.pending_orders),
                "current_positions": len(self.current_positions),
                "account_equity": self.account_equity,
                "risk_level": self.risk_level.value,
                "trading_allowed": self.halt_manager.is_trading_allowed()["trading_allowed"],
                "last_updated": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting execution summary: {e}")
            return {"error": str(e)}
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current position details."""
        return self.current_positions.copy()
    
    def get_pending_orders(self) -> Dict[str, Dict]:
        """Get pending order details."""
        return {
            order_id: {
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                "price": order.price,
                "created_at": order.created_at,
                "strategy": order.strategy_id
            }
            for order_id, order in self.pending_orders.items()
        }
    
    def update_account_equity(self, new_equity: float):
        """Update account equity for position sizing."""
        try:
            self.account_equity = new_equity
            self.halt_manager.update_equity(new_equity)
            logger.debug(f"Account equity updated to ${new_equity:,.2f}")
        except Exception as e:
            logger.error(f"Error updating account equity: {e}")
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order."""
        try:
            if order_id not in self.pending_orders:
                return {"status": "error", "message": "Order not found or not cancellable"}
            
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELED
            
            # Move to rejected orders for tracking
            self.rejected_orders[order_id] = order
            del self.pending_orders[order_id]
            
            logger.info(f"Order {order_id} cancelled successfully")
            
            return {
                "status": "success",
                "order_id": order_id,
                "message": "Order cancelled successfully"
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {"status": "error", "message": f"Cancellation error: {e}"}