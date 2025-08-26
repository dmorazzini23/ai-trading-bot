"""
Sophisticated Transaction Cost Analysis and Profit Validation.

This module provides comprehensive transaction cost modeling including
spread costs, market impact, commissions, and opportunity costs.
Validates that trades exceed total costs with required safety margins.
"""
from ai_trading.logging import get_logger
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any
logger = get_logger(__name__)
from ai_trading.core.constants import EXECUTION_PARAMETERS, RISK_PARAMETERS

def estimate_cost(quantity: float, price: float, bps: float=1.0) -> float:
    return quantity * price * (bps / 10000.0)

def _finite_nonneg(name: str, v: float | None) -> float:
    if v is None:
        raise ValueError(f'{name}_none')
    x = float(v)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError(f'{name}_invalid:{v}')
    return x

def _bounded_rate(name: str, v: float | None) -> float:
    x = _finite_nonneg(name, v)
    if x > 1.0:
        raise ValueError(f'{name}_gt1:{v}')
    return x

def _finite_pos(name: str, v: float | None) -> float:
    x = _finite_nonneg(name, v)
    if x <= 0.0:
        raise ValueError(f'{name}_invalid:{v}')
    return x

class TradeType(Enum):
    """Type of trade for cost calculation."""
    MARKET_ORDER = 'market'
    LIMIT_ORDER = 'limit'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    REBALANCE = 'rebalance'

class LiquidityTier(Enum):
    """Liquidity tier classification for impact modeling."""
    HIGH_LIQUIDITY = 'high'
    MEDIUM_LIQUIDITY = 'medium'
    LOW_LIQUIDITY = 'low'
    ILLIQUID = 'illiquid'

@dataclass
class TransactionCostBreakdown:
    """Detailed breakdown of transaction costs."""
    spread_cost: float
    commission: float
    market_impact: float
    opportunity_cost: float
    borrowing_cost: float
    total_cost: float
    cost_per_share: float
    cost_percentage: float

    def __post_init__(self):
        """Validate cost breakdown."""
        self.total_cost = self.spread_cost + self.commission + self.market_impact + self.opportunity_cost + self.borrowing_cost
        self.cost_percentage = max(0.0, min(1.0, self.cost_percentage))

@dataclass
class ProfitabilityAnalysis:
    """Analysis of trade profitability vs transaction costs."""
    expected_profit: float
    transaction_cost: float
    net_expected_profit: float
    profit_margin: float
    cost_ratio: float
    safety_margin: float
    is_profitable: bool
    confidence_level: float

class TransactionCostCalculator:
    """
    Sophisticated transaction cost calculator with market impact modeling.
    
    Provides detailed cost analysis including spread, commission, market impact,
    and opportunity costs. Validates trade profitability with safety margins.
    """

    def __init__(self, commission_rate: float=0.0001, min_commission: float=0.0, max_commission: float=1.0, safety_margin_multiplier: float=2.0):
        """
        Initialize transaction cost calculator.
        
        Args:
            commission_rate: Commission rate as decimal (e.g., 0.0001 = 1bp)
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade
            safety_margin_multiplier: Required safety margin over costs
        """
        self.commission_rate = _bounded_rate('commission_rate', commission_rate)
        self.min_commission = _finite_nonneg('min_commission', min_commission)
        self.max_commission = _finite_nonneg('max_commission', max_commission)
        self.safety_margin_multiplier = _finite_pos('safety_margin_multiplier', safety_margin_multiplier)
        self.limit_slippage = EXECUTION_PARAMETERS.get('LIMIT_ORDER_SLIPPAGE', 0.005)
        self.max_market_impact = EXECUTION_PARAMETERS.get('MAX_MARKET_IMPACT', 0.01)
        self.safety_margin_multiplier = RISK_PARAMETERS.get('SAFETY_MARGIN_MULTIPLIER', 2.0)
        self.impact_model_params = {'linear_coefficient': 0.0001, 'sqrt_coefficient': 0.001, 'permanent_ratio': 0.3, 'recovery_halflife': 300}
        logger.info(f'TransactionCostCalculator initialized with commission_rate={commission_rate:.4f}, safety_margin={safety_margin_multiplier}x')

    def calculate_spread_cost(self, symbol: str, trade_size: float, market_data: dict[str, Any]) -> float:
        """
        Calculate bid-ask spread cost for a trade.
        
        Args:
            symbol: Trading symbol
            trade_size: Number of shares to trade (positive for buy, negative for sell)
            market_data: Market data including bid/ask prices
            
        Returns:
            Spread cost in dollars
        """
        try:
            quotes = market_data.get('quotes', {})
            if symbol not in quotes:
                price = market_data.get('prices', {}).get(symbol, 100.0)
                estimated_spread_pct = self._estimate_spread_percentage(symbol, market_data)
                return abs(trade_size) * price * estimated_spread_pct / 2
            quote = quotes[symbol]
            bid = quote.get('bid', 0)
            ask = quote.get('ask', 0)
            if bid <= 0 or ask <= 0:
                price = market_data.get('prices', {}).get(symbol, 100.0)
                estimated_spread_pct = self._estimate_spread_percentage(symbol, market_data)
                return abs(trade_size) * price * estimated_spread_pct / 2
            spread = ask - bid
            (bid + ask) / 2
            spread_cost = abs(trade_size) * spread / 2
            logger.debug(f'Spread cost for {symbol}: ${spread_cost:.4f} (spread=${spread:.4f}, size={trade_size})')
            return spread_cost
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.error('SPREAD_COST_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            price = market_data.get('prices', {}).get(symbol, 100.0)
            return abs(trade_size) * price * 0.001

    def calculate_commission(self, symbol: str, trade_size: float, trade_value: float) -> float:
        """
        Calculate broker commission for a trade.
        
        Args:
            symbol: Trading symbol
            trade_size: Number of shares to trade
            trade_value: Dollar value of trade
            
        Returns:
            Commission cost in dollars
        """
        try:
            commission = trade_value * self.commission_rate
            commission = max(self.min_commission, min(self.max_commission, commission))
            logger.debug(f'Commission for {symbol}: ${commission:.4f} (value=${trade_value:.2f}, rate={self.commission_rate:.4f})')
            return commission
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.error('COMMISSION_CALC_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            return max(self.min_commission, trade_value * 0.0001)

    def calculate_market_impact(self, symbol: str, trade_size: float, market_data: dict[str, Any]) -> tuple[float, float]:
        """
        Calculate market impact cost using sophisticated modeling.
        
        Args:
            symbol: Trading symbol
            trade_size: Number of shares to trade
            market_data: Market data including volume, volatility
            
        Returns:
            Tuple of (temporary_impact, permanent_impact) in dollars
        """
        try:
            price = market_data.get('prices', {}).get(symbol, 100.0)
            volume_data = market_data.get('volumes', {})
            volatility_data = market_data.get('volatility', {})
            avg_daily_volume = volume_data.get(symbol, 1000000)
            volatility = volatility_data.get(symbol, 0.02)
            participation_rate = abs(trade_size) / max(avg_daily_volume, 1)
            linear_impact = self.impact_model_params['linear_coefficient'] * participation_rate
            sqrt_impact = self.impact_model_params['sqrt_coefficient'] * math.sqrt(participation_rate)
            vol_adjustment = volatility / 0.02
            temp_impact_pct = (linear_impact + sqrt_impact) * vol_adjustment
            temp_impact_pct = min(temp_impact_pct, self.max_market_impact)
            perm_impact_pct = temp_impact_pct * self.impact_model_params['permanent_ratio']
            trade_value = abs(trade_size) * price
            temporary_impact = trade_value * temp_impact_pct
            permanent_impact = trade_value * perm_impact_pct
            logger.debug(f'Market impact for {symbol}: temp=${temporary_impact:.4f}, perm=${permanent_impact:.4f} (participation={participation_rate:.4f})')
            return (temporary_impact, permanent_impact)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.error('MARKET_IMPACT_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            trade_value = abs(trade_size) * market_data.get('prices', {}).get(symbol, 100.0)
            impact = trade_value * 0.005
            return (impact * 0.7, impact * 0.3)

    def calculate_opportunity_cost(self, symbol: str, expected_delay: float, expected_return: float, trade_value: float) -> float:
        """
        Calculate opportunity cost from execution delay.
        
        Args:
            symbol: Trading symbol
            expected_delay: Expected execution delay in minutes
            expected_return: Expected return rate per day
            trade_value: Dollar value of trade
            
        Returns:
            Opportunity cost in dollars
        """
        try:
            if expected_delay <= 0 or expected_return == 0:
                return 0.0
            delay_fraction = expected_delay / 390.0
            opportunity_cost = delay_fraction * abs(expected_return) * trade_value
            logger.debug(f'Opportunity cost for {symbol}: ${opportunity_cost:.4f} (delay={expected_delay}min, return={expected_return:.4f})')
            return opportunity_cost
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.error('OPPORTUNITY_COST_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            return 0.0

    def calculate_borrowing_cost(self, symbol: str, trade_size: float, trade_value: float, holding_period_days: float=1.0) -> float:
        """
        Calculate borrowing cost for short selling.
        
        Args:
            symbol: Trading symbol
            trade_size: Number of shares (negative for short)
            trade_value: Dollar value of trade
            holding_period_days: Expected holding period
            
        Returns:
            Borrowing cost in dollars
        """
        try:
            if trade_size >= 0:
                return 0.0
            annual_borrow_rate = 0.01
            daily_rate = annual_borrow_rate / 365.0
            borrowing_cost = trade_value * daily_rate * holding_period_days
            logger.debug(f'Borrowing cost for {symbol}: ${borrowing_cost:.4f} (rate={annual_borrow_rate:.1%}, days={holding_period_days})')
            return borrowing_cost
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.error('BORROWING_COST_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            return 0.0

    def calculate_total_transaction_cost(self, symbol: str, trade_size: float, trade_type: TradeType, market_data: dict[str, Any], expected_delay: float=1.0, expected_return: float=0.0, holding_period_days: float=1.0) -> TransactionCostBreakdown:
        """
        Calculate comprehensive transaction cost breakdown.
        
        Args:
            symbol: Trading symbol
            trade_size: Number of shares to trade
            trade_type: Type of trade (market, limit, etc.)
            market_data: Complete market data
            expected_delay: Expected execution delay in minutes
            expected_return: Expected return rate
            holding_period_days: Expected holding period for short sales
            
        Returns:
            Detailed transaction cost breakdown
        """
        try:
            trade_size_abs = _finite_pos('qty', abs(trade_size))
            price = _finite_pos('price', market_data.get('prices', {}).get(symbol, 100.0))
            trade_value = trade_size_abs * price
            spread_cost = self.calculate_spread_cost(symbol, trade_size, market_data)
            commission = self.calculate_commission(symbol, trade_size, trade_value)
            temp_impact, perm_impact = self.calculate_market_impact(symbol, trade_size, market_data)
            market_impact = temp_impact + perm_impact
            opportunity_cost = self.calculate_opportunity_cost(symbol, expected_delay, expected_return, trade_value)
            borrowing_cost = self.calculate_borrowing_cost(symbol, trade_size, trade_value, holding_period_days)
            if trade_type == TradeType.LIMIT_ORDER:
                spread_cost *= 0.5
                opportunity_cost *= 1.5
            elif trade_type == TradeType.MARKET_ORDER:
                opportunity_cost *= 0.5
            total_cost = spread_cost + commission + market_impact + opportunity_cost + borrowing_cost
            cost_per_share = total_cost / trade_size_abs
            cost_percentage = total_cost / max(trade_value, 1)
            result = TransactionCostBreakdown(spread_cost=spread_cost, commission=commission, market_impact=market_impact, opportunity_cost=opportunity_cost, borrowing_cost=borrowing_cost, total_cost=total_cost, cost_per_share=cost_per_share, cost_percentage=cost_percentage)
            logger.info(f'Transaction cost for {symbol}: ${total_cost:.4f} ({cost_percentage:.4f}% of trade value)')
            logger.debug('TX_COSTS_COMPUTED', extra={'symbol': symbol, 'qty': trade_size_abs, 'price': price, 'fee_rate': None, 'slippage_bps': None, 'atr_used': None, 'result_keys': list(result.__dict__.keys())})
            return result
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.error('TX_COST_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            raise

    def validate_trade_profitability(self, symbol: str, trade_size: float, expected_profit: float, market_data: dict[str, Any], trade_type: TradeType=TradeType.LIMIT_ORDER, confidence_level: float=0.8) -> ProfitabilityAnalysis:
        """
        Validate that trade meets profitability requirements with safety margins.
        
        Args:
            symbol: Trading symbol
            trade_size: Number of shares to trade
            expected_profit: Expected profit in dollars
            market_data: Market data for cost calculation
            trade_type: Type of trade
            confidence_level: Confidence in profit estimate
            
        Returns:
            Comprehensive profitability analysis
        """
        try:
            cost_breakdown = self.calculate_total_transaction_cost(symbol, trade_size, trade_type, market_data)
            transaction_cost = cost_breakdown.total_cost
            net_expected_profit = expected_profit - transaction_cost
            profit_margin = net_expected_profit / max(abs(expected_profit), 0.01) if expected_profit != 0 else -1
            cost_ratio = transaction_cost / max(abs(expected_profit), 0.01) if expected_profit != 0 else float('inf')
            required_profit = transaction_cost * self.safety_margin_multiplier
            safety_margin = (expected_profit - required_profit) / max(required_profit, 0.01)
            is_profitable = expected_profit > required_profit and net_expected_profit > 0 and (cost_ratio < 0.5) and (confidence_level >= 0.6)
            logger.info(f'Profitability analysis for {symbol}: profit=${expected_profit:.4f}, cost=${transaction_cost:.4f}, net=${net_expected_profit:.4f}, profitable={is_profitable}')
            return ProfitabilityAnalysis(expected_profit=expected_profit, transaction_cost=transaction_cost, net_expected_profit=net_expected_profit, profit_margin=profit_margin, cost_ratio=cost_ratio, safety_margin=safety_margin, is_profitable=is_profitable, confidence_level=confidence_level)
        except ValueError as e:
            logger.warning('PROFITABILITY_VALIDATION_VALUE_ERROR', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            raise ValueError(f'Invalid trade parameters: {str(e)}')
        except KeyError as e:
            logger.warning('PROFITABILITY_VALIDATION_KEY_ERROR', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            raise KeyError(f'Required market data missing: {str(e)}')
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.error('PROFITABILITY_VALIDATION_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            return ProfitabilityAnalysis(expected_profit=expected_profit, transaction_cost=abs(expected_profit) * 0.5, net_expected_profit=expected_profit * 0.5, profit_margin=-0.5, cost_ratio=2.0, safety_margin=-1.0, is_profitable=False, confidence_level=confidence_level)

    def _estimate_spread_percentage(self, symbol: str, market_data: dict[str, Any]) -> float:
        """Estimate bid-ask spread percentage based on market data."""
        try:
            liquidity_tier = self._classify_liquidity(symbol, market_data)
            spread_estimates = {LiquidityTier.HIGH_LIQUIDITY: 0.0005, LiquidityTier.MEDIUM_LIQUIDITY: 0.002, LiquidityTier.LOW_LIQUIDITY: 0.005, LiquidityTier.ILLIQUID: 0.01}
            return spread_estimates.get(liquidity_tier, 0.005)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.warning('SPREAD_PERCENT_ESTIMATE_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            return 0.005

    def _classify_liquidity(self, symbol: str, market_data: dict[str, Any]) -> LiquidityTier:
        """Classify symbol liquidity based on volume and market cap."""
        try:
            volumes = market_data.get('volumes', {})
            avg_volume = volumes.get(symbol, 100000)
            if avg_volume >= 1000000:
                return LiquidityTier.HIGH_LIQUIDITY
            elif avg_volume >= 100000:
                return LiquidityTier.MEDIUM_LIQUIDITY
            elif avg_volume >= 10000:
                return LiquidityTier.LOW_LIQUIDITY
            else:
                return LiquidityTier.ILLIQUID
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            logger.warning('LIQUIDITY_CLASSIFICATION_FAILED', extra={'cause': e.__class__.__name__, 'detail': str(e), 'symbol': symbol})
            return LiquidityTier.MEDIUM_LIQUIDITY

def create_transaction_cost_calculator(config: dict[str, Any] | None=None) -> TransactionCostCalculator:
    """
    Factory function to create transaction cost calculator with configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured TransactionCostCalculator instance
    """
    if config is None:
        config = {}
    return TransactionCostCalculator(commission_rate=config.get('commission_rate', 0.0001), min_commission=config.get('min_commission', 0.0), max_commission=config.get('max_commission', 1.0), safety_margin_multiplier=config.get('safety_margin_multiplier', 2.0))