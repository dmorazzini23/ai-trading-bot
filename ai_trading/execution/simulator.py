"""
Fill simulation and slippage modeling for execution testing.

Provides realistic execution simulation including slippage,
partial fills, and market impact modeling.
"""

import logging
import math
import random

# Use the centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from ..core.enums import OrderSide, OrderType


class SlippageModel:
    """
    Market impact and slippage modeling.

    Models realistic slippage based on order size, market conditions,
    and execution urgency for backtesting and simulation.
    """

    def __init__(self):
        """Initialize slippage model."""
        # AI-AGENT-REF: Slippage and market impact modeling
        self.base_slippage_bps = 5  # Base slippage in basis points
        self.market_impact_factor = 0.1  # Market impact scaling
        self.volatility_factor = 1.0  # Current market volatility

        logger.info("SlippageModel initialized")

    def calculate_slippage(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType,
        **kwargs,
    ) -> float:
        """
        Calculate expected slippage for an order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Order price
            order_type: Type of order

        Returns:
            Slippage in dollars per share
        """
        try:
            # Base slippage from bid-ask spread
            base_slippage = price * (self.base_slippage_bps / 10000)

            # Market impact based on order size
            notional_value = quantity * price
            size_impact = self._calculate_size_impact(notional_value)

            # Urgency impact (market orders have higher slippage)
            urgency_impact = self._calculate_urgency_impact(order_type)

            # Volatility impact
            volatility_impact = base_slippage * self.volatility_factor

            # Side impact (buying typically has positive slippage, selling negative)
            side_multiplier = 1 if side == OrderSide.BUY else -1

            total_slippage = (
                base_slippage + size_impact + urgency_impact + volatility_impact
            ) * side_multiplier

            # Add random component for realism
            random_component = random.gauss(0, base_slippage * 0.2)
            total_slippage += random_component

            logger.debug(
                f"Slippage calculated for {symbol}: {total_slippage:.4f} per share"
            )

            return total_slippage

        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0.0

    def _calculate_size_impact(self, notional_value: float) -> float:
        """Calculate market impact based on order size."""
        # Square root model for market impact
        impact_factor = math.sqrt(notional_value / 1000000)  # Normalized to $1M
        return impact_factor * self.market_impact_factor

    def _calculate_urgency_impact(self, order_type: OrderType) -> float:
        """Calculate impact based on order urgency."""
        urgency_impacts = {
            OrderType.MARKET: 0.003,  # 3 bps additional for market orders
            OrderType.LIMIT: 0.0,  # No additional impact for limit orders
            OrderType.STOP: 0.002,  # 2 bps for stop orders
            OrderType.STOP_LIMIT: 0.001,  # 1 bp for stop-limit orders
        }
        return urgency_impacts.get(order_type, 0.0)

    def update_market_conditions(self, volatility: float, liquidity: float):
        """Update market conditions affecting slippage."""
        try:
            self.volatility_factor = max(0.5, min(2.0, volatility))

            # Adjust base slippage based on liquidity
            liquidity_factor = max(0.5, min(2.0, 1.0 / liquidity))
            self.base_slippage_bps = 5 * liquidity_factor

            logger.debug(
                f"Market conditions updated: volatility={volatility:.2f}, "
                f"liquidity={liquidity:.2f}"
            )

        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")


class FillSimulator:
    """
    Realistic order fill simulation.

    Simulates order execution with partial fills, delays,
    and realistic fill patterns for backtesting.
    """

    def __init__(self, slippage_model: SlippageModel = None):
        """Initialize fill simulator."""
        # AI-AGENT-REF: Order fill simulation
        self.slippage_model = slippage_model or SlippageModel()
        self.fill_probability = 0.95  # Base fill probability
        self.partial_fill_probability = 0.3  # Probability of partial fills
        self.max_fill_delay = 60  # Maximum fill delay in seconds

        logger.info("FillSimulator initialized")

    def simulate_fill(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType,
        **kwargs,
    ) -> dict:
        """
        Simulate order fill with realistic execution.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            order_type: Order type

        Returns:
            Fill simulation result dictionary
        """
        try:
            result = {
                "filled": False,
                "fill_quantity": 0,
                "fill_price": price,
                "slippage": 0.0,
                "fill_time": 0,
                "partial_fills": [],
                "rejection_reason": None,
            }

            # Check if order should be filled
            if not self._should_fill(order_type, price):
                result["rejection_reason"] = "Price limit not met"
                return result

            # Calculate slippage
            slippage = self.slippage_model.calculate_slippage(
                symbol, side, quantity, price, order_type
            )
            result["slippage"] = slippage
            result["fill_price"] = price + slippage

            # Determine if partial fill
            if self._should_partial_fill(quantity, order_type):
                fills = self._simulate_partial_fills(quantity, result["fill_price"])
                result["partial_fills"] = fills
                result["fill_quantity"] = sum(fill["quantity"] for fill in fills)
                result["filled"] = result["fill_quantity"] > 0
            else:
                # Full fill
                fill_time = self._calculate_fill_time(order_type)
                result["filled"] = True
                result["fill_quantity"] = quantity
                result["fill_time"] = fill_time
                result["partial_fills"] = [
                    {
                        "quantity": quantity,
                        "price": result["fill_price"],
                        "time": fill_time,
                    }
                ]

            logger.debug(
                f"Fill simulated: {symbol} {quantity}@{result['fill_price']:.2f}, "
                f"slippage={slippage:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error simulating fill: {e}")
            return {
                "filled": False,
                "fill_quantity": 0,
                "fill_price": price,
                "slippage": 0.0,
                "fill_time": 0,
                "partial_fills": [],
                "rejection_reason": f"Simulation error: {e}",
            }

    def _should_fill(self, order_type: OrderType, price: float) -> bool:
        """Determine if order should be filled."""
        # Market orders almost always fill
        if order_type == OrderType.MARKET:
            return random.random() < 0.98

        # Limit orders depend on price attractiveness
        # Simplified: assume all limit orders at market price fill
        return random.random() < self.fill_probability

    def _should_partial_fill(self, quantity: int, order_type: OrderType) -> bool:
        """Determine if order should have partial fills."""
        # Large orders more likely to have partial fills
        size_factor = min(1.0, quantity / 1000)  # Normalize to 1000 shares

        # Market orders less likely to have partials
        type_factor = 0.5 if order_type == OrderType.MARKET else 1.0

        partial_prob = self.partial_fill_probability * size_factor * type_factor
        return random.random() < partial_prob

    def _simulate_partial_fills(
        self, total_quantity: int, fill_price: float
    ) -> list[dict]:
        """Simulate sequence of partial fills."""
        fills = []
        remaining = total_quantity
        fill_time = 0

        while remaining > 0 and len(fills) < 5:  # Max 5 partial fills
            # Random fill size (20-80% of remaining)
            fill_pct = random.uniform(0.2, 0.8)
            fill_qty = max(1, int(remaining * fill_pct))
            fill_qty = min(fill_qty, remaining)

            # Small price variation for each fill
            price_variation = random.gauss(0, fill_price * 0.001)  # 0.1% variation
            actual_fill_price = fill_price + price_variation

            fill_time += random.randint(5, 30)  # 5-30 seconds between fills

            fills.append(
                {"quantity": fill_qty, "price": actual_fill_price, "time": fill_time}
            )

            remaining -= fill_qty

        return fills

    def _calculate_fill_time(self, order_type: OrderType) -> int:
        """Calculate realistic fill time."""
        base_times = {
            OrderType.MARKET: 5,  # 5 seconds average
            OrderType.LIMIT: 30,  # 30 seconds average
            OrderType.STOP: 15,  # 15 seconds average
            OrderType.STOP_LIMIT: 45,  # 45 seconds average
        }

        base_time = base_times.get(order_type, 30)

        # Add random variation
        variation = random.randint(-base_time // 2, base_time)
        fill_time = max(1, base_time + variation)

        return fill_time

    def update_market_conditions(self, volatility: float, volume: float):
        """Update simulation parameters based on market conditions."""
        try:
            # High volatility increases partial fill probability
            self.partial_fill_probability = 0.3 * (1 + volatility)

            # High volume improves fill probability
            volume_factor = min(2.0, volume)
            self.fill_probability = min(0.99, 0.95 * volume_factor)

            # Update slippage model
            liquidity = volume  # Simplified: volume proxy for liquidity
            self.slippage_model.update_market_conditions(volatility, liquidity)

            logger.debug(
                f"Fill simulator updated: vol={volatility:.2f}, vol={volume:.2f}"
            )

        except Exception as e:
            logger.error(f"Error updating fill simulator: {e}")
