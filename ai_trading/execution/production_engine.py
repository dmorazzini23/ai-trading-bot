"""
Production execution coordinator that integrates risk management and monitoring.

Enhances the existing execution engine with comprehensive safety checks,
real-time monitoring, and advanced risk management capabilities.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

# Use the centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from ..core.constants import EXECUTION_PARAMETERS
from ..core.enums import OrderSide, OrderType, RiskLevel
from ..monitoring import AlertManager, AlertSeverity
from ..risk import DynamicPositionSizer, RiskManager, TradingHaltManager
from .engine import ExecutionAlgorithm, Order, OrderStatus


class ExecutionResult:
    """
    Execution result class for tracking order execution outcomes.

    Provides comprehensive tracking of execution status, order details,
    execution timestamp, and any error messages or metadata.
    """

    def __init__(
        self,
        status: str,
        order_id: str,
        symbol: str,
        side: str | None = None,
        quantity: int | None = None,
        fill_price: float | None = None,
        message: str = "",
        execution_time: datetime | None = None,
        **kwargs,
    ):
        """Initialize execution result."""
        # AI-AGENT-REF: Execution result tracking for order outcomes
        self.status = status  # success, failed, rejected, etc.
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.fill_price = fill_price
        self.message = message
        self.execution_time = execution_time or datetime.now(UTC)

        # Additional metadata
        self.actual_slippage_bps = kwargs.get("actual_slippage_bps", 0.0)
        self.execution_time_ms = kwargs.get("execution_time_ms", 0.0)
        self.notional_value = kwargs.get("notional_value", 0.0)
        self.error_code = kwargs.get("error_code")
        self.venue = kwargs.get("venue", "simulation")

        # Track timestamp using UTC as per AGENTS.md
        self.timestamp = datetime.now(UTC)

        logger.debug(
            f"ExecutionResult created: {self.status} for order {self.order_id}"
        )

    @property
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status.lower() == "success"

    @property
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status.lower() in ["failed", "error", "rejected"]

    @property
    def is_partial(self) -> bool:
        """Check if execution was partial."""
        return self.status.lower() in ["partial", "partially_filled"]

    def to_dict(self) -> dict[str, Any]:
        """Convert execution result to dictionary representation."""
        return {
            "status": self.status,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "message": self.message,
            "execution_time": (
                self.execution_time.isoformat() if self.execution_time else None
            ),
            "timestamp": self.timestamp.isoformat(),
            "actual_slippage_bps": self.actual_slippage_bps,
            "execution_time_ms": self.execution_time_ms,
            "notional_value": self.notional_value,
            "error_code": self.error_code,
            "venue": self.venue,
            "is_successful": self.is_successful,
            "is_failed": self.is_failed,
            "is_partial": self.is_partial,
        }

    def __str__(self) -> str:
        """String representation of execution result."""
        return f"ExecutionResult({self.status}: {self.order_id} {self.symbol})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ExecutionResult(status='{self.status}', order_id='{self.order_id}', "
            f"symbol='{self.symbol}', quantity={self.quantity}, "
            f"fill_price={self.fill_price})"
        )


class OrderRequest:
    """
    Order request class for encapsulating order parameters with validation.

    Provides order parameter validation, serialization capabilities for API requests,
    and integration with the existing execution system.
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        strategy: str = "unknown",
        time_in_force: str = "DAY",
        **kwargs,
    ):
        """Initialize order request with validation."""
        # AI-AGENT-REF: Order request with validation and serialization
        self.symbol = symbol.upper() if symbol else ""
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.strategy = strategy
        self.time_in_force = time_in_force

        # Additional parameters
        self.client_order_id = kwargs.get("client_order_id", f"req_{int(time.time())}")
        self.stop_price = kwargs.get("stop_price")
        self.target_price = kwargs.get("target_price")
        self.min_quantity = kwargs.get("min_quantity", 0)
        self.max_participation_rate = kwargs.get("max_participation_rate", 0.1)
        self.urgency_level = kwargs.get("urgency_level", "normal")
        self.notes = kwargs.get("notes", "")

        # Risk parameters
        self.max_slippage_bps = kwargs.get("max_slippage_bps", 50)
        self.position_size_limit = kwargs.get("position_size_limit")

        # Metadata
        self.created_at = datetime.now(UTC)
        self.source_system = kwargs.get("source_system", "ai_trading")
        self.request_id = kwargs.get(
            "request_id", f"req_{self.created_at.strftime('%Y%m%d_%H%M%S')}"
        )

        # Validate the request upon creation
        self._validation_errors = []
        self._is_valid = self._validate()

        logger.debug(f"OrderRequest created: {self.side} {self.quantity} {self.symbol}")

    def _validate(self) -> bool:
        """Validate order request parameters."""
        self._validation_errors = []

        # Symbol validation
        if not self.symbol or len(self.symbol) < 1:
            self._validation_errors.append("Symbol is required and cannot be empty")

        # Quantity validation
        if self.quantity <= 0:
            self._validation_errors.append("Quantity must be positive")

        if self.quantity > 1000000:  # Reasonable upper limit
            self._validation_errors.append("Quantity exceeds maximum limit")

        # Side validation
        if not isinstance(self.side, OrderSide):
            self._validation_errors.append("Side must be a valid OrderSide enum")

        # Order type validation
        if not isinstance(self.order_type, OrderType):
            self._validation_errors.append("Order type must be a valid OrderType enum")

        # Price validation for limit orders
        if self.order_type == OrderType.LIMIT:
            if not self.price or self.price <= 0:
                self._validation_errors.append("Limit orders require a valid price")

        # Stop price validation
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if not self.stop_price or self.stop_price <= 0:
                self._validation_errors.append("Stop orders require a valid stop price")

        # Risk parameter validation
        if self.max_slippage_bps < 0 or self.max_slippage_bps > 1000:
            self._validation_errors.append(
                "Max slippage must be between 0 and 1000 basis points"
            )

        if self.max_participation_rate <= 0 or self.max_participation_rate > 1:
            self._validation_errors.append(
                "Max participation rate must be between 0 and 1"
            )

        return len(self._validation_errors) == 0

    @property
    def is_valid(self) -> bool:
        """Check if order request is valid."""
        return self._is_valid

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors."""
        return self._validation_errors.copy()

    @property
    def notional_value(self) -> float:
        """Calculate notional value of the order."""
        price = self.price or 100.0  # Default price for market orders
        return abs(self.quantity * price)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate order request and return result with errors."""
        self._is_valid = self._validate()
        return self._is_valid, self.validation_errors

    def to_dict(self) -> dict[str, Any]:
        """Convert order request to dictionary for API serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, OrderSide) else self.side,
            "quantity": self.quantity,
            "order_type": (
                self.order_type.value
                if isinstance(self.order_type, OrderType)
                else self.order_type
            ),
            "price": self.price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "time_in_force": self.time_in_force,
            "client_order_id": self.client_order_id,
            "strategy": self.strategy,
            "min_quantity": self.min_quantity,
            "max_participation_rate": self.max_participation_rate,
            "urgency_level": self.urgency_level,
            "max_slippage_bps": self.max_slippage_bps,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "source_system": self.source_system,
            "request_id": self.request_id,
            "notional_value": self.notional_value,
            "is_valid": self.is_valid,
        }

    def to_api_request(self, broker_format: str = "alpaca") -> dict[str, Any]:
        """Convert to broker-specific API request format."""
        if broker_format.lower() == "alpaca":
            return {
                "symbol": self.symbol,
                "side": self.side.value,
                "type": self.order_type.value,
                "qty": str(self.quantity),
                "time_in_force": self.time_in_force,
                "client_order_id": self.client_order_id,
            }
        else:
            # Generic format
            return self.to_dict()

    def copy(self, **updates) -> "OrderRequest":
        """Create a copy of the order request with optional updates."""
        # Copy the original attributes, preserving enums
        kwargs = {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "price": self.price,
            "strategy": self.strategy,
            "time_in_force": self.time_in_force,
            "client_order_id": self.client_order_id,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "min_quantity": self.min_quantity,
            "max_participation_rate": self.max_participation_rate,
            "urgency_level": self.urgency_level,
            "notes": self.notes,
            "max_slippage_bps": self.max_slippage_bps,
            "position_size_limit": self.position_size_limit,
            "source_system": self.source_system,
        }

        # Apply updates
        kwargs.update(updates)

        # Create new instance with potentially new client_order_id
        if "client_order_id" not in updates:
            # Generate a new unique client_order_id for the copy
            import time

            kwargs["client_order_id"] = (
                f"req_{int(time.time() * 1000)}"  # Use milliseconds for uniqueness
            )

        return OrderRequest(**kwargs)

    def __str__(self) -> str:
        """String representation of order request."""
        return f"OrderRequest({self.side.value} {self.quantity} {self.symbol} @ {self.order_type.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"OrderRequest(symbol='{self.symbol}', side={self.side}, "
            f"quantity={self.quantity}, order_type={self.order_type}, "
            f"price={self.price}, valid={self.is_valid})"
        )


class ProductionExecutionCoordinator:
    """
    Production execution coordinator with comprehensive safety integration.

    Wraps the core execution engine with advanced risk management,
    monitoring, and safety mechanisms for production trading.
    """

    def __init__(
        self, account_equity: float, risk_level: RiskLevel = RiskLevel.MODERATE
    ):
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
            "last_reset": datetime.now(UTC),
        }

        # Position tracking
        self.current_positions = {}

        logger.info(
            f"ProductionExecutionCoordinator initialized with equity=${account_equity:,.2f}"
        )

    async def submit_order_request(
        self, order_request: OrderRequest
    ) -> ExecutionResult:
        """
        Submit order using OrderRequest object with comprehensive safety checks.

        Args:
            order_request: OrderRequest object containing order parameters

        Returns:
            ExecutionResult object with execution outcome
        """
        try:
            # Validate order request
            if not order_request.is_valid:
                error_msg = f"Invalid order request: {'; '.join(order_request.validation_errors)}"
                logger.warning(error_msg)
                return ExecutionResult(
                    status="rejected",
                    order_id=order_request.client_order_id,
                    symbol=order_request.symbol,
                    side=(
                        order_request.side.value
                        if isinstance(order_request.side, OrderSide)
                        else order_request.side
                    ),
                    quantity=order_request.quantity,
                    message=error_msg,
                )

            # Submit using the existing submit_order method
            return await self.submit_order(
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                order_type=order_request.order_type,
                price=order_request.price,
                strategy=order_request.strategy,
                metadata={
                    "client_order_id": order_request.client_order_id,
                    "time_in_force": order_request.time_in_force,
                    "stop_price": order_request.stop_price,
                    "target_price": order_request.target_price,
                    "min_quantity": order_request.min_quantity,
                    "max_participation_rate": order_request.max_participation_rate,
                    "urgency_level": order_request.urgency_level,
                    "notes": order_request.notes,
                    "source_system": order_request.source_system,
                    "request_id": order_request.request_id,
                },
            )

        except Exception as e:
            logger.error(f"Error submitting order request: {e}")
            return ExecutionResult(
                status="error",
                order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=(
                    order_request.side.value
                    if isinstance(order_request.side, OrderSide)
                    else order_request.side
                ),
                quantity=order_request.quantity,
                message=f"Submission error: {e}",
            )

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        strategy: str = "unknown",
        metadata: dict = None,
    ) -> ExecutionResult:
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
                **metadata or {},
            )

            # Step 2: Pre-execution safety checks
            safety_result = await self._comprehensive_safety_check(order)
            if not safety_result["approved"]:
                await self._handle_order_rejection(order, safety_result["reason"])
                return self._create_order_result(
                    order, "rejected", safety_result["reason"]
                )

            # Step 3: Position sizing optimization
            sizing_result = await self._optimize_order_size(order)
            if sizing_result["final_quantity"] == 0:
                await self._handle_order_rejection(
                    order, "Position sizing resulted in zero quantity"
                )
                return self._create_order_result(
                    order, "rejected", "Invalid position size"
                )

            # Update order with optimized quantity
            original_quantity = order.quantity
            order.quantity = sizing_result["final_quantity"]

            # Step 4: Market impact analysis
            impact_analysis = await self._analyze_market_impact(order)

            # Step 5: Execute order
            execution_result = await self._execute_order_with_monitoring(
                order, impact_analysis
            )

            # Step 6: Post-execution processing
            await self._post_execution_processing(
                order, execution_result, original_quantity
            )

            # Update statistics
            execution_time_ms = (time.time() - start_time) * 1000
            await self._update_execution_statistics(execution_result, execution_time_ms)

            return execution_result

        except Exception as e:
            logger.error(f"Error in order submission: {e}")
            return ExecutionResult(
                status="error",
                order_id="unknown",
                symbol=symbol,
                side=side.value if isinstance(side, OrderSide) else side,
                quantity=quantity,
                message=f"Submission error: {e}",
                error_code="submission_error",
            )

    async def _comprehensive_safety_check(self, order: Order) -> dict[str, Any]:
        """Perform comprehensive safety checks before execution."""
        try:
            # Check trading halt status
            halt_status = self.halt_manager.is_trading_allowed()
            if not halt_status["trading_allowed"]:
                return {
                    "approved": False,
                    "reason": f"Trading halted: {', '.join(halt_status['reasons'])}",
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
                position_history,
            )

            if not risk_assessment["approved"]:
                return {
                    "approved": False,
                    "reason": f"Risk assessment failed: {', '.join(risk_assessment['warnings'])}",
                }

            # Check for recent similar orders (prevent duplicate submissions)
            if self._has_recent_similar_order(order):
                return {"approved": False, "reason": "Similar order recently submitted"}

            return {"approved": True, "reason": "All safety checks passed"}

        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return {"approved": False, "reason": f"Safety check error: {e}"}

    async def _optimize_order_size(self, order: Order) -> dict[str, Any]:
        """Optimize order size using dynamic position sizing."""
        try:
            # Get market data (simplified for this implementation)
            market_data = {
                "current_price": order.price or 100.0,
                "atr": 2.0,  # Default ATR
                "volume": 1000000,  # Default volume
            }

            # Get historical data
            historical_data = {"returns": [], "trade_history": []}

            # Calculate optimal position size
            sizing_result = self.position_sizer.calculate_optimal_position(
                order.symbol,
                self.account_equity,
                market_data["current_price"],
                market_data,
                historical_data,
            )

            # Apply any halt manager position size multipliers
            halt_status = self.halt_manager.is_trading_allowed()
            position_multiplier = halt_status.get("position_size_multiplier", 1.0)

            # Calculate final quantity
            final_quantity = min(
                order.quantity,
                sizing_result["recommended_size"],
                int(order.quantity * position_multiplier),
            )

            return {
                "original_quantity": order.quantity,
                "recommended_quantity": sizing_result["recommended_size"],
                "final_quantity": max(0, final_quantity),
                "position_multiplier": position_multiplier,
                "sizing_warnings": sizing_result.get("warnings", []),
            }

        except Exception as e:
            logger.error(f"Error optimizing order size: {e}")
            return {"final_quantity": 0}

    async def _analyze_market_impact(self, order: Order) -> dict[str, Any]:
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
                "recommended_algorithm": self._recommend_execution_algorithm(
                    impact_level
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing market impact: {e}")
            return {"impact_level": "unknown", "estimated_slippage_bps": 0}

    async def _execute_order_with_monitoring(
        self, order: Order, impact_analysis: dict
    ) -> ExecutionResult:
        """Execute order with real-time monitoring."""
        try:
            # Add to pending orders
            self.pending_orders[order.id] = order

            # Simulate order execution (in production, integrate with actual broker API)
            await asyncio.sleep(0.1)  # Simulate execution delay

            # Simulate fill
            fill_price = order.price or 100.0
            if order.side == OrderSide.BUY:
                fill_price *= 1 + impact_analysis["estimated_slippage_bps"] / 10000
            else:
                fill_price *= 1 - impact_analysis["estimated_slippage_bps"] / 10000

            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.executed_at = datetime.now(UTC)

            # Move to completed orders
            self.completed_orders[order.id] = order
            del self.pending_orders[order.id]

            # Update position tracking
            self._update_position_tracking(order)

            # Calculate actual slippage
            expected_price = order.price or fill_price
            actual_slippage_bps = (
                abs(fill_price - expected_price) / expected_price * 10000
            )

            return ExecutionResult(
                status="success",
                order_id=order.id,
                symbol=order.symbol,
                side=(
                    order.side.value
                    if isinstance(order.side, OrderSide)
                    else order.side
                ),
                quantity=order.quantity,
                fill_price=fill_price,
                execution_time=order.executed_at,
                message=f"Order executed successfully at ${fill_price:.2f}",
                actual_slippage_bps=actual_slippage_bps,
                notional_value=order.quantity * fill_price,
            )

        except Exception as e:
            logger.error(f"Error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            self.rejected_orders[order.id] = order
            if order.id in self.pending_orders:
                del self.pending_orders[order.id]

            return ExecutionResult(
                status="failed",
                order_id=order.id,
                symbol=order.symbol,
                side=(
                    order.side.value
                    if isinstance(order.side, OrderSide)
                    else order.side
                ),
                quantity=order.quantity,
                message=f"Execution failed: {e}",
                error_code="execution_error",
            )

    async def _post_execution_processing(
        self, order: Order, execution_result: dict, original_quantity: int
    ):
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
                            "slippage_bps": execution_result.get(
                                "actual_slippage_bps", 0
                            ),
                        },
                        AlertSeverity.INFO,
                    )

                # Check for excessive slippage
                slippage = execution_result.get("actual_slippage_bps", 0)
                if slippage > EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"]:
                    await self.alert_manager.send_performance_alert(
                        "Excessive Slippage",
                        slippage,
                        EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"],
                        AlertSeverity.WARNING,
                    )

                logger.info(
                    f"Order {order.id} executed successfully: {order.symbol} "
                    f"{order.side.value} {order.quantity} @ ${execution_result['fill_price']:.2f}"
                )

            else:
                # Handle execution failure
                await self.alert_manager.send_trading_alert(
                    "Order Execution Failed",
                    order.symbol,
                    {
                        "order_id": order.id,
                        "reason": execution_result.get("message", "Unknown"),
                        "quantity": order.quantity,
                    },
                    AlertSeverity.WARNING,
                )

                logger.warning(
                    f"Order {order.id} execution failed: {execution_result.get('message')}"
                )

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
                {"order_id": order.id, "reason": reason, "quantity": order.quantity},
                AlertSeverity.WARNING,
            )

            logger.warning(f"Order {order.id} rejected: {reason}")

        except Exception as e:
            logger.error(f"Error handling order rejection: {e}")

    def _create_order_result(
        self, order: Order, status: str, message: str
    ) -> ExecutionResult:
        """Create standardized order result."""
        return ExecutionResult(
            status=status,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value if isinstance(order.side, OrderSide) else order.side,
            quantity=order.quantity,
            message=message,
            execution_time=datetime.now(UTC),
        )

    def _update_position_tracking(self, order: Order):
        """Update internal position tracking."""
        try:
            symbol = order.symbol
            quantity = (
                order.quantity if order.side == OrderSide.BUY else -order.quantity
            )
            fill_price = order.average_fill_price

            if symbol in self.current_positions:
                current_pos = self.current_positions[symbol]
                new_quantity = current_pos["quantity"] + quantity

                if new_quantity == 0:
                    del self.current_positions[symbol]
                else:
                    # Calculate new average price
                    total_cost = (
                        current_pos["quantity"] * current_pos["avg_price"]
                        + quantity * fill_price
                    )
                    new_avg_price = total_cost / new_quantity

                    self.current_positions[symbol] = {
                        "quantity": new_quantity,
                        "avg_price": new_avg_price,
                        "last_updated": datetime.now(UTC),
                    }
            else:
                if quantity != 0:
                    self.current_positions[symbol] = {
                        "quantity": quantity,
                        "avg_price": fill_price,
                        "last_updated": datetime.now(UTC),
                    }

        except Exception as e:
            logger.error(f"Error updating position tracking: {e}")

    async def _update_execution_statistics(
        self, execution_result: dict, execution_time_ms: float
    ):
        """Update execution performance statistics."""
        try:
            self.execution_stats["total_orders"] += 1

            if execution_result["status"] == "success":
                self.execution_stats["successful_orders"] += 1

                # Update average execution time
                alpha = 0.1
                if self.execution_stats["average_execution_time_ms"] == 0:
                    self.execution_stats["average_execution_time_ms"] = (
                        execution_time_ms
                    )
                else:
                    self.execution_stats["average_execution_time_ms"] = (
                        alpha * execution_time_ms
                        + (1 - alpha)
                        * self.execution_stats["average_execution_time_ms"]
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
            cutoff_time = datetime.now(UTC) - timedelta(seconds=30)

            for existing_order in self.pending_orders.values():
                if (
                    existing_order.symbol == order.symbol
                    and existing_order.side == order.side
                    and existing_order.quantity == order.quantity
                    and existing_order.created_at >= cutoff_time
                ):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking for similar orders: {e}")
            return False

    def _get_symbol_history(self, symbol: str) -> list[dict]:
        """Get trading history for a symbol."""
        # In production, this would query historical data
        return []

    def get_execution_summary(self) -> dict[str, Any]:
        """Get execution engine summary statistics."""
        try:
            success_rate = (
                self.execution_stats["successful_orders"]
                / self.execution_stats["total_orders"]
                * 100
                if self.execution_stats["total_orders"] > 0
                else 0
            )

            avg_slippage = (
                self.execution_stats["total_slippage_bps"]
                / self.execution_stats["successful_orders"]
                if self.execution_stats["successful_orders"] > 0
                else 0
            )

            return {
                "execution_stats": self.execution_stats,
                "success_rate_pct": success_rate,
                "average_slippage_bps": avg_slippage,
                "pending_orders": len(self.pending_orders),
                "current_positions": len(self.current_positions),
                "account_equity": self.account_equity,
                "risk_level": self.risk_level.value,
                "trading_allowed": self.halt_manager.is_trading_allowed()[
                    "trading_allowed"
                ],
                "last_updated": datetime.now(UTC),
            }

        except Exception as e:
            logger.error(f"Error getting execution summary: {e}")
            return {"error": str(e)}

    def get_current_positions(self) -> dict[str, dict]:
        """Get current position details."""
        return self.current_positions.copy()

    def get_pending_orders(self) -> dict[str, dict]:
        """Get pending order details."""
        return {
            order_id: {
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                "price": order.price,
                "created_at": order.created_at,
                "strategy": order.strategy_id,
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

    async def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancel a pending order."""
        try:
            if order_id not in self.pending_orders:
                return ExecutionResult(
                    status="error",
                    order_id=order_id,
                    symbol="unknown",
                    message="Order not found or not cancellable",
                )

            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELED

            # Move to rejected orders for tracking
            self.rejected_orders[order_id] = order
            del self.pending_orders[order_id]

            logger.info(f"Order {order_id} cancelled successfully")

            return ExecutionResult(
                status="success",
                order_id=order_id,
                symbol=order.symbol,
                side=(
                    order.side.value
                    if isinstance(order.side, OrderSide)
                    else order.side
                ),
                quantity=order.quantity,
                message="Order cancelled successfully",
            )

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return ExecutionResult(
                status="error",
                order_id=order_id,
                symbol="unknown",
                message=f"Cancellation error: {e}",
                error_code="cancellation_error",
            )
