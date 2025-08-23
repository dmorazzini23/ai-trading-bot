"""
Fill simulation and slippage modeling for execution testing.

Provides realistic execution simulation including slippage,
partial fills, and market impact modeling.
"""
import math
import random
from ai_trading.logging import logger
from ..core.enums import OrderSide, OrderType

class SlippageModel:
    """
    Market impact and slippage modeling.

    Models realistic slippage based on order size, market conditions,
    and execution urgency for backtesting and simulation.
    """

    def __init__(self):
        """Initialize slippage model."""
        self.base_slippage_bps = 5
        self.market_impact_factor = 0.1
        self.volatility_factor = 1.0
        logger.info('SlippageModel initialized')

    def calculate_slippage(self, symbol: str, side: OrderSide, quantity: int, price: float, order_type: OrderType, **kwargs) -> float:
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
            base_slippage = price * (self.base_slippage_bps / 10000)
            notional_value = quantity * price
            size_impact = self._calculate_size_impact(notional_value)
            urgency_impact = self._calculate_urgency_impact(order_type)
            volatility_impact = base_slippage * self.volatility_factor
            side_multiplier = 1 if side == OrderSide.BUY else -1
            total_slippage = (base_slippage + size_impact + urgency_impact + volatility_impact) * side_multiplier
            random_component = random.gauss(0, base_slippage * 0.2)
            total_slippage += random_component
            logger.debug(f'Slippage calculated for {symbol}: {total_slippage:.4f} per share')
            return total_slippage
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error(f'Error calculating slippage: {e}')
            return 0.0

    def _calculate_size_impact(self, notional_value: float) -> float:
        """Calculate market impact based on order size."""
        impact_factor = math.sqrt(notional_value / 1000000)
        return impact_factor * self.market_impact_factor

    def _calculate_urgency_impact(self, order_type: OrderType) -> float:
        """Calculate impact based on order urgency."""
        urgency_impacts = {OrderType.MARKET: 0.003, OrderType.LIMIT: 0.0, OrderType.STOP: 0.002, OrderType.STOP_LIMIT: 0.001}
        return urgency_impacts.get(order_type, 0.0)

    def update_market_conditions(self, volatility: float, liquidity: float):
        """Update market conditions affecting slippage."""
        try:
            self.volatility_factor = max(0.5, min(2.0, volatility))
            liquidity_factor = max(0.5, min(2.0, 1.0 / liquidity))
            self.base_slippage_bps = 5 * liquidity_factor
            logger.debug(f'Market conditions updated: volatility={volatility:.2f}, liquidity={liquidity:.2f}')
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error(f'Error updating market conditions: {e}')

class FillSimulator:
    """
    Realistic order fill simulation.

    Simulates order execution with partial fills, delays,
    and realistic fill patterns for backtesting.
    """

    def __init__(self, slippage_model: SlippageModel=None):
        """Initialize fill simulator."""
        self.slippage_model = slippage_model or SlippageModel()
        self.fill_probability = 0.95
        self.partial_fill_probability = 0.3
        self.max_fill_delay = 60
        logger.info('FillSimulator initialized')

    def simulate_fill(self, symbol: str, side: OrderSide, quantity: int, price: float, order_type: OrderType, **kwargs) -> dict:
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
            result = {'filled': False, 'fill_quantity': 0, 'fill_price': price, 'slippage': 0.0, 'fill_time': 0, 'partial_fills': [], 'rejection_reason': None}
            if not self._should_fill(order_type, price):
                result['rejection_reason'] = 'Price limit not met'
                return result
            slippage = self.slippage_model.calculate_slippage(symbol, side, quantity, price, order_type)
            result['slippage'] = slippage
            result['fill_price'] = price + slippage
            if self._should_partial_fill(quantity, order_type):
                fills = self._simulate_partial_fills(quantity, result['fill_price'])
                result['partial_fills'] = fills
                result['fill_quantity'] = sum((fill['quantity'] for fill in fills))
                result['filled'] = result['fill_quantity'] > 0
            else:
                fill_time = self._calculate_fill_time(order_type)
                result['filled'] = True
                result['fill_quantity'] = quantity
                result['fill_time'] = fill_time
                result['partial_fills'] = [{'quantity': quantity, 'price': result['fill_price'], 'time': fill_time}]
            logger.debug(f"Fill simulated: {symbol} {quantity}@{result['fill_price']:.2f}, slippage={slippage:.4f}")
            return result
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error(f'Error simulating fill: {e}')
            return {'filled': False, 'fill_quantity': 0, 'fill_price': price, 'slippage': 0.0, 'fill_time': 0, 'partial_fills': [], 'rejection_reason': f'Simulation error: {e}'}

    def _should_fill(self, order_type: OrderType, price: float) -> bool:
        """Determine if order should be filled."""
        if order_type == OrderType.MARKET:
            return random.random() < 0.98
        return random.random() < self.fill_probability

    def _should_partial_fill(self, quantity: int, order_type: OrderType) -> bool:
        """Determine if order should have partial fills."""
        size_factor = min(1.0, quantity / 1000)
        type_factor = 0.5 if order_type == OrderType.MARKET else 1.0
        partial_prob = self.partial_fill_probability * size_factor * type_factor
        return random.random() < partial_prob

    def _simulate_partial_fills(self, total_quantity: int, fill_price: float) -> list[dict]:
        """Simulate sequence of partial fills."""
        fills = []
        remaining = total_quantity
        fill_time = 0
        while remaining > 0 and len(fills) < 5:
            fill_pct = random.uniform(0.2, 0.8)
            fill_qty = max(1, int(remaining * fill_pct))
            fill_qty = min(fill_qty, remaining)
            price_variation = random.gauss(0, fill_price * 0.001)
            actual_fill_price = fill_price + price_variation
            fill_time += random.randint(5, 30)
            fills.append({'quantity': fill_qty, 'price': actual_fill_price, 'time': fill_time})
            remaining -= fill_qty
        return fills

    def _calculate_fill_time(self, order_type: OrderType) -> int:
        """Calculate realistic fill time."""
        base_times = {OrderType.MARKET: 5, OrderType.LIMIT: 30, OrderType.STOP: 15, OrderType.STOP_LIMIT: 45}
        base_time = base_times.get(order_type, 30)
        variation = random.randint(-base_time // 2, base_time)
        fill_time = max(1, base_time + variation)
        return fill_time

    def update_market_conditions(self, volatility: float, volume: float):
        """Update simulation parameters based on market conditions."""
        try:
            self.partial_fill_probability = 0.3 * (1 + volatility)
            volume_factor = min(2.0, volume)
            self.fill_probability = min(0.99, 0.95 * volume_factor)
            liquidity = volume
            self.slippage_model.update_market_conditions(volatility, liquidity)
            logger.debug(f'Fill simulator updated: vol={volatility:.2f}, vol={volume:.2f}')
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error(f'Error updating fill simulator: {e}')