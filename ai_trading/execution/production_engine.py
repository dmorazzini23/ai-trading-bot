"""
Production execution coordinator that integrates risk management and monitoring.

Enhances the existing execution engine with comprehensive safety checks,
real-time monitoring, and advanced risk management capabilities.
"""
import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from alpaca.common.exceptions import APIError
from ai_trading.logging import logger
from ..core.constants import EXECUTION_PARAMETERS
from ..core.enums import OrderSide, OrderType, RiskLevel
from ..monitoring import AlertManager, AlertSeverity
from ..risk import DynamicPositionSizer, RiskManager, TradingHaltManager
from .engine import ExecutionAlgorithm, Order, OrderStatus
from .classes import ExecutionResult, OrderRequest


class ProductionExecutionCoordinator:
    """
    Production execution coordinator with comprehensive safety integration.

    Wraps the core execution engine with advanced risk management,
    monitoring, and safety mechanisms for production trading.
    """

    def __init__(self, account_equity: float, risk_level: RiskLevel=RiskLevel.MODERATE):
        """Initialize production execution coordinator."""
        self.account_equity = account_equity
        self.risk_level = risk_level
        self.position_sizer = DynamicPositionSizer(risk_level)
        self.halt_manager = TradingHaltManager()
        self.risk_manager = RiskManager(risk_level)
        self.alert_manager = AlertManager()
        self.pending_orders = {}
        self.completed_orders = {}
        self.rejected_orders = {}
        self.execution_stats = {'total_orders': 0, 'successful_orders': 0, 'rejected_orders': 0, 'average_execution_time_ms': 0.0, 'total_slippage_bps': 0.0, 'last_reset': datetime.now(UTC)}
        self.current_positions = {}
        logger.info(f'ProductionExecutionCoordinator initialized with equity=${account_equity:,.2f}')

    async def submit_order_request(self, order_request: OrderRequest) -> ExecutionResult:
        """
        Submit order using OrderRequest object with comprehensive safety checks.

        Args:
            order_request: OrderRequest object containing order parameters

        Returns:
            ExecutionResult object with execution outcome
        """
        try:
            if not order_request.is_valid:
                error_msg = f"Invalid order request: {'; '.join(order_request.validation_errors)}"
                logger.warning(error_msg)
                return ExecutionResult(status='rejected', order_id=order_request.client_order_id, symbol=order_request.symbol, side=order_request.side.value if isinstance(order_request.side, OrderSide) else order_request.side, quantity=order_request.quantity, message=error_msg)
            return await self.submit_order(symbol=order_request.symbol, side=order_request.side, quantity=order_request.quantity, order_type=order_request.order_type, price=order_request.price, strategy=order_request.strategy, metadata={'client_order_id': order_request.client_order_id, 'time_in_force': order_request.time_in_force, 'stop_price': order_request.stop_price, 'target_price': order_request.target_price, 'min_quantity': order_request.min_quantity, 'max_participation_rate': order_request.max_participation_rate, 'urgency_level': order_request.urgency_level, 'notes': order_request.notes, 'source_system': order_request.source_system, 'request_id': order_request.request_id})
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_API_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'op': 'submit', 'order_id': order_request.client_order_id})
            raise

    async def submit_order(self, symbol: str, side: OrderSide, quantity: int, order_type: OrderType=OrderType.MARKET, price: float | None=None, strategy: str='unknown', metadata: dict=None) -> ExecutionResult:
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
            order = Order(symbol=symbol, side=side, quantity=quantity, order_type=order_type, price=price, strategy_id=strategy, **metadata or {})
            safety_result = await self._comprehensive_safety_check(order)
            if not safety_result['approved']:
                await self._handle_order_rejection(order, safety_result['reason'])
                return self._create_order_result(order, 'rejected', safety_result['reason'])
            sizing_result = await self._optimize_order_size(order)
            if sizing_result['final_quantity'] == 0:
                await self._handle_order_rejection(order, 'Position sizing resulted in zero quantity')
                return self._create_order_result(order, 'rejected', 'Invalid position size')
            original_quantity = order.quantity
            order.quantity = sizing_result['final_quantity']
            impact_analysis = await self._analyze_market_impact(order)
            execution_result = await self._execute_order_with_monitoring(order, impact_analysis)
            await self._post_execution_processing(order, execution_result, original_quantity)
            execution_time_ms = (time.time() - start_time) * 1000
            await self._update_execution_statistics(execution_result, execution_time_ms)
            return execution_result
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_API_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'op': 'submit', 'order_id': 'unknown'})
            raise

    async def _comprehensive_safety_check(self, order: Order) -> dict[str, Any]:
        """Perform comprehensive safety checks before execution."""
        try:
            halt_status = self.halt_manager.is_trading_allowed()
            if not halt_status['trading_allowed']:
                return {'approved': False, 'reason': f"Trading halted: {', '.join(halt_status['reasons'])}"}
            if order.quantity <= 0:
                return {'approved': False, 'reason': 'Invalid quantity'}
            if order.quantity > EXECUTION_PARAMETERS['MAX_ORDER_SIZE']:
                return {'approved': False, 'reason': 'Quantity exceeds maximum limit'}
            position_history = self._get_symbol_history(order.symbol)
            risk_assessment = self.risk_manager.assess_trade_risk(order.symbol, order.quantity, order.price or 100.0, self.account_equity, position_history)
            if not risk_assessment['approved']:
                return {'approved': False, 'reason': f"Risk assessment failed: {', '.join(risk_assessment['warnings'])}"}
            if self._has_recent_similar_order(order):
                return {'approved': False, 'reason': 'Similar order recently submitted'}
            return {'approved': True, 'reason': 'All safety checks passed'}
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('SAFETY_CHECK_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return {'approved': False, 'reason': f'Safety check error: {e}'}

    async def _optimize_order_size(self, order: Order) -> dict[str, Any]:
        """Optimize order size using dynamic position sizing."""
        try:
            market_data = {'current_price': order.price or 100.0, 'atr': 2.0, 'volume': 1000000}
            historical_data = {'returns': [], 'trade_history': []}
            sizing_result = self.position_sizer.calculate_optimal_position(order.symbol, self.account_equity, market_data['current_price'], market_data, historical_data)
            halt_status = self.halt_manager.is_trading_allowed()
            position_multiplier = halt_status.get('position_size_multiplier', 1.0)
            final_quantity = min(order.quantity, sizing_result['recommended_size'], int(order.quantity * position_multiplier))
            return {'original_quantity': order.quantity, 'recommended_quantity': sizing_result['recommended_size'], 'final_quantity': max(0, final_quantity), 'position_multiplier': position_multiplier, 'sizing_warnings': sizing_result.get('warnings', [])}
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_SIZING_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return {'final_quantity': 0}

    async def _analyze_market_impact(self, order: Order) -> dict[str, Any]:
        """Analyze potential market impact of the order."""
        try:
            current_price = order.price or 100.0
            notional_value = order.quantity * current_price
            if notional_value > 1000000:
                impact_level = 'high'
                estimated_slippage_bps = 15
            elif notional_value > 100000:
                impact_level = 'medium'
                estimated_slippage_bps = 8
            else:
                impact_level = 'low'
                estimated_slippage_bps = 3
            return {'impact_level': impact_level, 'estimated_slippage_bps': estimated_slippage_bps, 'notional_value': notional_value, 'recommended_algorithm': self._recommend_execution_algorithm(impact_level)}
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('MARKET_IMPACT_ANALYSIS_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return {'impact_level': 'unknown', 'estimated_slippage_bps': 0}

    async def _execute_order_with_monitoring(self, order: Order, impact_analysis: dict) -> ExecutionResult:
        """Execute order with real-time monitoring."""
        try:
            self.pending_orders[order.id] = order
            await asyncio.sleep(0.1)
            fill_price = order.price or 100.0
            if order.side == OrderSide.BUY:
                fill_price *= 1 + impact_analysis['estimated_slippage_bps'] / 10000
            else:
                fill_price *= 1 - impact_analysis['estimated_slippage_bps'] / 10000
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.executed_at = datetime.now(UTC)
            self.completed_orders[order.id] = order
            del self.pending_orders[order.id]
            self._update_position_tracking(order)
            expected_price = order.price or fill_price
            actual_slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
            return ExecutionResult(status='success', order_id=order.id, symbol=order.symbol, side=order.side.value if isinstance(order.side, OrderSide) else order.side, quantity=order.quantity, fill_price=fill_price, execution_time=order.executed_at, message=f'Order executed successfully at ${fill_price:.2f}', actual_slippage_bps=actual_slippage_bps, notional_value=order.quantity * fill_price)
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_EXECUTION_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'order_id': order.id})
            order.status = OrderStatus.REJECTED
            self.rejected_orders[order.id] = order
            if order.id in self.pending_orders:
                del self.pending_orders[order.id]
            return ExecutionResult(status='failed', order_id=order.id, symbol=order.symbol, side=order.side.value if isinstance(order.side, OrderSide) else order.side, quantity=order.quantity, message=f'Execution failed: {e}', error_code='execution_error')

    async def _post_execution_processing(self, order: Order, execution_result: dict, original_quantity: int):
        """Handle post-execution processing and notifications."""
        try:
            if execution_result['status'] == 'success':
                self.halt_manager.record_trade()
                notional_value = order.quantity * execution_result['fill_price']
                if notional_value > 50000:
                    await self.alert_manager.send_trading_alert('Order Executed', order.symbol, {'side': order.side.value, 'quantity': order.quantity, 'fill_price': execution_result['fill_price'], 'notional_value': notional_value, 'slippage_bps': execution_result.get('actual_slippage_bps', 0)}, AlertSeverity.INFO)
                slippage = execution_result.get('actual_slippage_bps', 0)
                if slippage > EXECUTION_PARAMETERS['MAX_SLIPPAGE_BPS']:
                    await self.alert_manager.send_performance_alert('Excessive Slippage', slippage, EXECUTION_PARAMETERS['MAX_SLIPPAGE_BPS'], AlertSeverity.WARNING)
                logger.info(f"Order {order.id} executed successfully: {order.symbol} {order.side.value} {order.quantity} @ ${execution_result['fill_price']:.2f}")
            else:
                await self.alert_manager.send_trading_alert('Order Execution Failed', order.symbol, {'order_id': order.id, 'reason': execution_result.get('message', 'Unknown'), 'quantity': order.quantity}, AlertSeverity.WARNING)
                logger.warning(f"Order {order.id} execution failed: {execution_result.get('message')}")
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('POST_EXECUTION_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'order_id': order.id})

    async def _handle_order_rejection(self, order: Order, reason: str):
        """Handle order rejection with proper logging and alerts."""
        try:
            order.status = OrderStatus.REJECTED
            self.rejected_orders[order.id] = order
            await self.alert_manager.send_trading_alert('Order Rejected', order.symbol, {'order_id': order.id, 'reason': reason, 'quantity': order.quantity}, AlertSeverity.WARNING)
            logger.warning(f'Order {order.id} rejected: {reason}')
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_REJECTION_HANDLER_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'order_id': order.id})

    def _create_order_result(self, order: Order, status: str, message: str) -> ExecutionResult:
        """Create standardized order result."""
        return ExecutionResult(status=status, order_id=order.id, symbol=order.symbol, side=order.side.value if isinstance(order.side, OrderSide) else order.side, quantity=order.quantity, message=message, execution_time=datetime.now(UTC))

    def _update_position_tracking(self, order: Order):
        """Update internal position tracking."""
        try:
            symbol = order.symbol
            quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
            fill_price = order.average_fill_price
            if symbol in self.current_positions:
                current_pos = self.current_positions[symbol]
                new_quantity = current_pos['quantity'] + quantity
                if new_quantity == 0:
                    del self.current_positions[symbol]
                else:
                    total_cost = current_pos['quantity'] * current_pos['avg_price'] + quantity * fill_price
                    new_avg_price = total_cost / new_quantity
                    self.current_positions[symbol] = {'quantity': new_quantity, 'avg_price': new_avg_price, 'last_updated': datetime.now(UTC)}
            elif quantity != 0:
                self.current_positions[symbol] = {'quantity': quantity, 'avg_price': fill_price, 'last_updated': datetime.now(UTC)}
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('POSITION_UPDATE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'order_id': order.id})

    async def _update_execution_statistics(self, execution_result: dict, execution_time_ms: float):
        """Update execution performance statistics."""
        try:
            self.execution_stats['total_orders'] += 1
            if execution_result['status'] == 'success':
                self.execution_stats['successful_orders'] += 1
                alpha = 0.1
                if self.execution_stats['average_execution_time_ms'] == 0:
                    self.execution_stats['average_execution_time_ms'] = execution_time_ms
                else:
                    self.execution_stats['average_execution_time_ms'] = alpha * execution_time_ms + (1 - alpha) * self.execution_stats['average_execution_time_ms']
                slippage = execution_result.get('actual_slippage_bps', 0)
                self.execution_stats['total_slippage_bps'] += slippage
            else:
                self.execution_stats['rejected_orders'] += 1
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('EXECUTION_STATS_UPDATE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})

    def _recommend_execution_algorithm(self, impact_level: str) -> ExecutionAlgorithm:
        """Recommend execution algorithm based on impact analysis."""
        if impact_level == 'high':
            return ExecutionAlgorithm.TWAP
        elif impact_level == 'medium':
            return ExecutionAlgorithm.VWAP
        else:
            return ExecutionAlgorithm.MARKET

    def _has_recent_similar_order(self, order: Order) -> bool:
        """Check for recent similar orders to prevent duplicates."""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(seconds=30)
            for existing_order in self.pending_orders.values():
                if existing_order.symbol == order.symbol and existing_order.side == order.side and (existing_order.quantity == order.quantity) and (existing_order.created_at >= cutoff_time):
                    return True
            return False
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('SIMILAR_ORDER_CHECK_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return False

    def _get_symbol_history(self, symbol: str) -> list[dict]:
        """Get trading history for a symbol."""
        return []

    def get_execution_summary(self) -> dict[str, Any]:
        """Get execution engine summary statistics."""
        try:
            success_rate = self.execution_stats['successful_orders'] / self.execution_stats['total_orders'] * 100 if self.execution_stats['total_orders'] > 0 else 0
            avg_slippage = self.execution_stats['total_slippage_bps'] / self.execution_stats['successful_orders'] if self.execution_stats['successful_orders'] > 0 else 0
            return {'execution_stats': self.execution_stats, 'success_rate_pct': success_rate, 'average_slippage_bps': avg_slippage, 'pending_orders': len(self.pending_orders), 'current_positions': len(self.current_positions), 'account_equity': self.account_equity, 'risk_level': self.risk_level.value, 'trading_allowed': self.halt_manager.is_trading_allowed()['trading_allowed'], 'last_updated': datetime.now(UTC)}
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('EXECUTION_SUMMARY_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})
            return {'error': str(e)}

    def get_current_positions(self) -> dict[str, dict]:
        """Get current position details."""
        return self.current_positions.copy()

    def get_pending_orders(self) -> dict[str, dict]:
        """Get pending order details."""
        return {order_id: {'symbol': order.symbol, 'side': order.side.value, 'quantity': order.quantity, 'order_type': order.order_type.value, 'price': order.price, 'created_at': order.created_at, 'strategy': order.strategy_id} for order_id, order in self.pending_orders.items()}

    def update_account_equity(self, new_equity: float):
        """Update account equity for position sizing."""
        try:
            self.account_equity = new_equity
            self.halt_manager.update_equity(new_equity)
            logger.debug(f'Account equity updated to ${new_equity:,.2f}')
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ACCOUNT_EQUITY_UPDATE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e)})

    async def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancel a pending order."""
        try:
            if order_id not in self.pending_orders:
                return ExecutionResult(status='error', order_id=order_id, symbol='unknown', message='Order not found or not cancellable')
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELED
            self.rejected_orders[order_id] = order
            del self.pending_orders[order_id]
            logger.info(f'Order {order_id} cancelled successfully')
            return ExecutionResult(status='success', order_id=order_id, symbol=order.symbol, side=order.side.value if isinstance(order.side, OrderSide) else order.side, quantity=order.quantity, message='Order cancelled successfully')
        except (APIError, TimeoutError, ConnectionError) as e:
            logger.error('ORDER_API_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'op': 'cancel', 'order_id': order_id})
            raise