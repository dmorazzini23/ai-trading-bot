"""
Enhanced order policy with marketable limit orders and smart routing.

Provides intelligent order placement with symbol-specific parameters,
IOC for fades, and market fallback logic.
"""
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from ai_trading.execution.costs import get_symbol_costs
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = 'market'
    LIMIT = 'limit'
    MARKETABLE_LIMIT = 'marketable_limit'
    IOC = 'ioc'
    GTD = 'gtd'

class OrderUrgency(Enum):
    """Order urgency levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    URGENT = 'urgent'

@dataclass
class OrderParameters:
    """Symbol-specific order parameters."""
    symbol: str
    spread_multiplier: float = 1.0
    ioc_threshold_bps: float = 5.0
    market_fallback_bps: float = 20.0
    max_wait_seconds: float = 30.0
    min_fill_ratio: float = 0.8
    high_volume_threshold: float = 2.0
    high_volume_multiplier: float = 1.5

@dataclass
class MarketData:
    """Current market data for order placement."""
    symbol: str
    bid: float
    ask: float
    mid: float
    spread_bps: float
    volume_ratio: float = 1.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    @property
    def half_spread(self) -> float:
        """Half spread in dollars."""
        return (self.ask - self.bid) / 2

    @property
    def is_wide_spread(self) -> bool:
        """Check if spread is unusually wide."""
        return self.spread_bps > 10.0

class SmartOrderRouter:
    """
    Smart order routing with adaptive order types and timing.
    """

    def __init__(self):
        """Initialize smart order router."""
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._symbol_params: dict[str, OrderParameters] = {}
        self._active_orders: dict[str, dict] = {}

    def get_order_params(self, symbol: str) -> OrderParameters:
        """
        Get order parameters for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            OrderParameters for the symbol
        """
        if symbol not in self._symbol_params:
            self._symbol_params[symbol] = OrderParameters(symbol=symbol)
        return self._symbol_params[symbol]

    def update_order_params(self, symbol: str, **param_updates) -> None:
        """
        Update order parameters for symbol.

        Args:
            symbol: Trading symbol
            **param_updates: Parameter updates
        """
        params = self.get_order_params(symbol)
        for key, value in param_updates.items():
            if hasattr(params, key):
                setattr(params, key, value)
            else:
                self.logger.warning(f'Unknown parameter: {key}')

    def calculate_limit_price(self, market_data: MarketData, side: str, urgency: OrderUrgency=OrderUrgency.MEDIUM) -> tuple[float, OrderType]:
        """
        Calculate optimal limit price and order type.

        Args:
            market_data: Current market data
            side: Order side ('buy' or 'sell')
            urgency: Order urgency level

        Returns:
            Tuple of (limit_price, recommended_order_type)
        """
        params = self.get_order_params(market_data.symbol)
        get_symbol_costs(market_data.symbol)
        k = params.spread_multiplier
        urgency_multipliers = {OrderUrgency.LOW: 0.5, OrderUrgency.MEDIUM: 1.0, OrderUrgency.HIGH: 1.5, OrderUrgency.URGENT: 2.0}
        k *= urgency_multipliers.get(urgency, 1.0)
        if market_data.volume_ratio > params.high_volume_threshold:
            k *= params.high_volume_multiplier
            self.logger.info(f'High volume detected for {market_data.symbol} ({market_data.volume_ratio:.1f}x), adjusting spread')
        half_spread = market_data.half_spread
        limit_offset = k * half_spread
        if side.lower() == 'buy':
            limit_price = market_data.bid + limit_offset
            limit_price = min(limit_price, market_data.mid)
        else:
            limit_price = market_data.ask - limit_offset
            limit_price = max(limit_price, market_data.mid)
        recommended_type = self._recommend_order_type(market_data, params, urgency)
        return (limit_price, recommended_type)

    def _recommend_order_type(self, market_data: MarketData, params: OrderParameters, urgency: OrderUrgency) -> OrderType:
        """Recommend order type based on market conditions."""
        spread_bps = market_data.spread_bps
        if urgency == OrderUrgency.URGENT and spread_bps > params.market_fallback_bps:
            return OrderType.MARKET
        if spread_bps > params.ioc_threshold_bps or urgency in [OrderUrgency.HIGH, OrderUrgency.URGENT]:
            return OrderType.IOC
        return OrderType.MARKETABLE_LIMIT

    def create_order_request(self, symbol: str, side: str, quantity: float, market_data: MarketData, urgency: OrderUrgency=OrderUrgency.MEDIUM, custom_params: dict | None=None) -> dict:
        """
        Create optimized order request.

        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            market_data: Current market data
            urgency: Order urgency
            custom_params: Custom order parameters

        Returns:
            Order request dict
        """
        limit_price, order_type = self.calculate_limit_price(market_data, side, urgency)
        order_request = {'symbol': symbol, 'side': side.lower(), 'quantity': abs(quantity), 'type': order_type.value, 'limit_price': limit_price, 'urgency': urgency.value, 'created_at': datetime.now(UTC).isoformat()}
        params = self.get_order_params(symbol)
        if order_type == OrderType.IOC:
            order_request.update({'time_in_force': 'IOC', 'allow_partial_fill': True, 'min_fill_ratio': params.min_fill_ratio})
        elif order_type == OrderType.MARKETABLE_LIMIT:
            order_request.update({'time_in_force': 'DAY', 'allow_partial_fill': True, 'max_wait_seconds': params.max_wait_seconds})
        elif order_type == OrderType.MARKET:
            order_request.update({'type': 'market', 'allow_partial_fill': False})
        if custom_params:
            order_request.update(custom_params)
        costs = get_symbol_costs(symbol)
        position_value = abs(quantity) * market_data.mid
        cost_estimate = costs.total_execution_cost_bps(market_data.volume_ratio)
        order_request['cost_estimate'] = {'cost_bps': cost_estimate, 'cost_dollars': position_value * (cost_estimate / 10000), 'slippage_risk': 'high' if market_data.volume_ratio > 2.0 else 'normal'}
        self.logger.info(f'Created {order_type.value} order for {symbol}: {side} {quantity} @ ${limit_price:.4f} (est. cost: {cost_estimate:.1f}bps)')
        return order_request

    def should_cancel_and_retry(self, order_id: str, order_info: dict, current_market: MarketData) -> tuple[bool, str]:
        """
        Determine if order should be cancelled and retried.

        Args:
            order_id: Order ID
            order_info: Current order information
            current_market: Current market data

        Returns:
            Tuple of (should_cancel, reason)
        """
        if order_id not in self._active_orders:
            return (False, 'Order not tracked')
        order_data = self._active_orders[order_id]
        params = self.get_order_params(order_data['symbol'])
        order_age = (datetime.now(UTC) - order_data['created_at']).total_seconds()
        if order_age > params.max_wait_seconds:
            return (True, f'Order aged out ({order_age:.1f}s > {params.max_wait_seconds}s)')
        original_limit = order_data.get('limit_price', 0)
        current_mid = current_market.mid
        if original_limit > 0:
            price_deviation = abs(current_mid - original_limit) / original_limit
            if price_deviation > 0.005:
                return (True, f'Market moved significantly ({price_deviation:.2%})')
        filled_qty = order_info.get('filled_quantity', 0)
        total_qty = order_info.get('quantity', 1)
        fill_ratio = filled_qty / total_qty
        if fill_ratio > 0 and fill_ratio < params.min_fill_ratio:
            if order_age > params.max_wait_seconds / 2:
                return (True, f'Poor fill ratio ({fill_ratio:.1%}) after {order_age:.1f}s')
        return (False, 'Order should continue')

    def handle_order_fade(self, original_order: dict, market_data: MarketData) -> dict:
        """
        Handle order fade by creating replacement order.

        Args:
            original_order: Original order that faded
            market_data: Current market data

        Returns:
            New order request
        """
        symbol = original_order['symbol']
        side = original_order['side']
        quantity = original_order['quantity']
        original_urgency = OrderUrgency(original_order.get('urgency', 'medium'))
        new_urgency = OrderUrgency.HIGH if original_urgency == OrderUrgency.MEDIUM else OrderUrgency.URGENT
        self.logger.warning(f'Order faded for {symbol}, creating replacement with urgency {new_urgency.value}')
        return self.create_order_request(symbol=symbol, side=side, quantity=quantity, market_data=market_data, urgency=new_urgency)

    def track_order(self, order_id: str, order_request: dict) -> None:
        """Track active order for monitoring."""
        self._active_orders[order_id] = {**order_request, 'created_at': datetime.now(UTC)}

    def untrack_order(self, order_id: str) -> None:
        """Remove order from tracking."""
        if order_id in self._active_orders:
            del self._active_orders[order_id]

    def get_routing_summary(self) -> dict:
        """Get summary of order routing activity."""
        return {'active_orders': len(self._active_orders), 'tracked_symbols': len(self._symbol_params), 'default_params': {'spread_multiplier': 1.0, 'ioc_threshold_bps': 5.0, 'market_fallback_bps': 20.0}}
_global_router: SmartOrderRouter | None = None

def get_smart_router() -> SmartOrderRouter:
    """Get or create global smart order router."""
    global _global_router
    if _global_router is None:
        _global_router = SmartOrderRouter()
    return _global_router

def create_smart_order(symbol: str, side: str, quantity: float, bid: float, ask: float, volume_ratio: float=1.0, urgency: str='medium') -> dict:
    """
    Convenience function to create smart order.

    Args:
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        bid: Current bid price
        ask: Current ask price
        volume_ratio: Volume ratio
        urgency: Order urgency

    Returns:
        Order request dict
    """
    router = get_smart_router()
    mid = (bid + ask) / 2
    spread_bps = (ask - bid) / mid * 10000 if mid > 0 else 0
    market_data = MarketData(symbol=symbol, bid=bid, ask=ask, mid=mid, spread_bps=spread_bps, volume_ratio=volume_ratio)
    urgency_enum = OrderUrgency(urgency.lower())
    return router.create_order_request(symbol=symbol, side=side, quantity=quantity, market_data=market_data, urgency=urgency_enum)