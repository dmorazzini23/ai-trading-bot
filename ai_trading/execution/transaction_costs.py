"""
Sophisticated Transaction Cost Analysis and Profit Validation.

This module provides comprehensive transaction cost modeling including
spread costs, market impact, commissions, and opportunity costs.
Validates that trades exceed total costs with required safety margins.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)  # AI-AGENT-REF: module logger

from ai_trading.core.constants import (  # AI-AGENT-REF: direct import without shim
    EXECUTION_PARAMETERS,
    RISK_PARAMETERS,
)


# AI-AGENT-REF: simple transaction cost estimator for tests
def estimate_cost(quantity: float, price: float, bps: float = 1.0) -> float:
    return quantity * price * (bps / 10_000.0)


def _finite_nonneg(name: str, v: float | None) -> float:  # AI-AGENT-REF: input guard helper
    if v is None:
        raise ValueError(f"{name}_none")
    x = float(v)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError(f"{name}_invalid:{v}")
    return x


def _bounded_rate(name: str, v: float | None) -> float:  # AI-AGENT-REF: bounded rate guard
    x = _finite_nonneg(name, v)
    if x > 1.0:
        raise ValueError(f"{name}_gt1:{v}")
    return x


def _finite_pos(name: str, v: float | None) -> float:  # AI-AGENT-REF: positive guard
    x = _finite_nonneg(name, v)
    if x <= 0.0:
        raise ValueError(f"{name}_invalid:{v}")
    return x


class TradeType(Enum):
    """Type of trade for cost calculation."""
    MARKET_ORDER = "market"
    LIMIT_ORDER = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    REBALANCE = "rebalance"


class LiquidityTier(Enum):
    """Liquidity tier classification for impact modeling."""
    HIGH_LIQUIDITY = "high"      # Large cap, high volume
    MEDIUM_LIQUIDITY = "medium"  # Mid cap, medium volume
    LOW_LIQUIDITY = "low"        # Small cap, low volume
    ILLIQUID = "illiquid"        # Very low volume


@dataclass
class TransactionCostBreakdown:
    """Detailed breakdown of transaction costs."""
    spread_cost: float           # Bid-ask spread cost
    commission: float            # Broker commission
    market_impact: float         # Market impact cost
    opportunity_cost: float      # Delay/timing opportunity cost
    borrowing_cost: float        # Short selling borrowing cost
    total_cost: float           # Total transaction cost
    cost_per_share: float       # Cost per share
    cost_percentage: float      # Cost as percentage of trade value

    def __post_init__(self):
        """Validate cost breakdown."""
        self.total_cost = (self.spread_cost + self.commission + self.market_impact +
                          self.opportunity_cost + self.borrowing_cost)
        self.cost_percentage = max(0.0, min(1.0, self.cost_percentage))


@dataclass
class ProfitabilityAnalysis:
    """Analysis of trade profitability vs transaction costs."""
    expected_profit: float       # Expected profit from trade
    transaction_cost: float      # Total transaction cost
    net_expected_profit: float   # Expected profit minus costs
    profit_margin: float         # Net profit margin percentage
    cost_ratio: float           # Transaction cost / expected profit ratio
    safety_margin: float        # Safety margin above costs
    is_profitable: bool         # Whether trade meets profitability threshold
    confidence_level: float     # Confidence in profit estimates


class TransactionCostCalculator:
    """
    Sophisticated transaction cost calculator with market impact modeling.
    
    Provides detailed cost analysis including spread, commission, market impact,
    and opportunity costs. Validates trade profitability with safety margins.
    """

    def __init__(self,
                 commission_rate: float = 0.0001,  # 1 bp commission
                 min_commission: float = 0.0,      # No minimum commission
                 max_commission: float = 1.0,      # $1 max commission
                 safety_margin_multiplier: float = 2.0):  # 2x safety margin
        """
        Initialize transaction cost calculator.
        
        Args:
            commission_rate: Commission rate as decimal (e.g., 0.0001 = 1bp)
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade
            safety_margin_multiplier: Required safety margin over costs
        """
        self.commission_rate = _bounded_rate("commission_rate", commission_rate)  # AI-AGENT-REF: validate rate
        self.min_commission = _finite_nonneg("min_commission", min_commission)  # AI-AGENT-REF: validate min
        self.max_commission = _finite_nonneg("max_commission", max_commission)  # AI-AGENT-REF: validate max
        self.safety_margin_multiplier = _finite_pos("safety_margin_multiplier", safety_margin_multiplier)  # AI-AGENT-REF: validate safety margin

        # Load enhanced parameters
        self.limit_slippage = EXECUTION_PARAMETERS.get('LIMIT_ORDER_SLIPPAGE', 0.005)
        self.max_market_impact = EXECUTION_PARAMETERS.get('MAX_MARKET_IMPACT', 0.01)
        self.safety_margin_multiplier = RISK_PARAMETERS.get('SAFETY_MARGIN_MULTIPLIER', 2.0)

        # Market impact model parameters
        self.impact_model_params = {
            'linear_coefficient': 0.0001,    # Linear impact coefficient
            'sqrt_coefficient': 0.001,       # Square root impact coefficient
            'permanent_ratio': 0.3,          # Ratio of permanent to temporary impact
            'recovery_halflife': 300         # Impact recovery half-life in seconds
        }

        _log.info(f"TransactionCostCalculator initialized with commission_rate={commission_rate:.4f}, "
                   f"safety_margin={safety_margin_multiplier}x")

    def calculate_spread_cost(self,
                            symbol: str,
                            trade_size: float,
                            market_data: dict[str, Any]) -> float:
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
                # Fallback: estimate spread from price
                price = market_data.get('prices', {}).get(symbol, 100.0)
                estimated_spread_pct = self._estimate_spread_percentage(symbol, market_data)
                return abs(trade_size) * price * estimated_spread_pct / 2

            quote = quotes[symbol]
            bid = quote.get('bid', 0)
            ask = quote.get('ask', 0)

            if bid <= 0 or ask <= 0:
                # Use fallback method
                price = market_data.get('prices', {}).get(symbol, 100.0)
                estimated_spread_pct = self._estimate_spread_percentage(symbol, market_data)
                return abs(trade_size) * price * estimated_spread_pct / 2

            spread = ask - bid
            mid_price = (bid + ask) / 2

            # Spread cost is half-spread times trade size
            spread_cost = abs(trade_size) * spread / 2

            _log.debug(f"Spread cost for {symbol}: ${spread_cost:.4f} "
                        f"(spread=${spread:.4f}, size={trade_size})")

            return spread_cost

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.error(
                "SPREAD_COST_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            # Conservative fallback
            price = market_data.get('prices', {}).get(symbol, 100.0)
            return abs(trade_size) * price * 0.001  # 10 bps fallback

    def calculate_commission(self,
                           symbol: str,
                           trade_size: float,
                           trade_value: float) -> float:
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
            # Calculate percentage-based commission
            commission = trade_value * self.commission_rate

            # Apply min/max constraints
            commission = max(self.min_commission, min(self.max_commission, commission))

            _log.debug(f"Commission for {symbol}: ${commission:.4f} "
                        f"(value=${trade_value:.2f}, rate={self.commission_rate:.4f})")

            return commission

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.error(
                "COMMISSION_CALC_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            return max(self.min_commission, trade_value * 0.0001)  # 1bp fallback

    def calculate_market_impact(self,
                              symbol: str,
                              trade_size: float,
                              market_data: dict[str, Any]) -> tuple[float, float]:
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

            # Get average daily volume
            avg_daily_volume = volume_data.get(symbol, 1000000)  # Default 1M shares

            # Get volatility
            volatility = volatility_data.get(symbol, 0.02)  # Default 2% daily vol

            # Participation rate (trade size as % of daily volume)
            participation_rate = abs(trade_size) / max(avg_daily_volume, 1)

            # Market impact model: combination of linear and square root terms
            linear_impact = self.impact_model_params['linear_coefficient'] * participation_rate
            sqrt_impact = self.impact_model_params['sqrt_coefficient'] * math.sqrt(participation_rate)

            # Volatility adjustment
            vol_adjustment = volatility / 0.02  # Normalize to 2% baseline

            # Total temporary impact percentage
            temp_impact_pct = (linear_impact + sqrt_impact) * vol_adjustment
            temp_impact_pct = min(temp_impact_pct, self.max_market_impact)  # Cap at max

            # Permanent impact (fraction of temporary)
            perm_impact_pct = temp_impact_pct * self.impact_model_params['permanent_ratio']

            # Convert to dollar amounts
            trade_value = abs(trade_size) * price
            temporary_impact = trade_value * temp_impact_pct
            permanent_impact = trade_value * perm_impact_pct

            _log.debug(f"Market impact for {symbol}: temp=${temporary_impact:.4f}, "
                        f"perm=${permanent_impact:.4f} (participation={participation_rate:.4f})")

            return temporary_impact, permanent_impact

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.error(
                "MARKET_IMPACT_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            # Conservative fallback
            trade_value = abs(trade_size) * market_data.get('prices', {}).get(symbol, 100.0)
            impact = trade_value * 0.005  # 50 bps fallback
            return impact * 0.7, impact * 0.3  # Split temp/perm

    def calculate_opportunity_cost(self,
                                 symbol: str,
                                 expected_delay: float,
                                 expected_return: float,
                                 trade_value: float) -> float:
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

            # Convert delay to fraction of trading day (6.5 hours = 390 minutes)
            delay_fraction = expected_delay / 390.0

            # Opportunity cost = delay * expected return * trade value
            opportunity_cost = delay_fraction * abs(expected_return) * trade_value

            _log.debug(f"Opportunity cost for {symbol}: ${opportunity_cost:.4f} "
                        f"(delay={expected_delay}min, return={expected_return:.4f})")

            return opportunity_cost

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.error(
                "OPPORTUNITY_COST_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            return 0.0

    def calculate_borrowing_cost(self,
                               symbol: str,
                               trade_size: float,
                               trade_value: float,
                               holding_period_days: float = 1.0) -> float:
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
            if trade_size >= 0:  # Not a short sale
                return 0.0

            # Simplified borrowing cost model
            # In practice, this would use real-time borrow rates
            annual_borrow_rate = 0.01  # 1% annual borrow rate (conservative)

            # Calculate cost for holding period
            daily_rate = annual_borrow_rate / 365.0
            borrowing_cost = trade_value * daily_rate * holding_period_days

            _log.debug(f"Borrowing cost for {symbol}: ${borrowing_cost:.4f} "
                        f"(rate={annual_borrow_rate:.1%}, days={holding_period_days})")

            return borrowing_cost

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.error(
                "BORROWING_COST_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            return 0.0

    def calculate_total_transaction_cost(self,
                                       symbol: str,
                                       trade_size: float,
                                       trade_type: TradeType,
                                       market_data: dict[str, Any],
                                       expected_delay: float = 1.0,
                                       expected_return: float = 0.0,
                                       holding_period_days: float = 1.0) -> TransactionCostBreakdown:
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
            trade_size_abs = _finite_pos("qty", abs(trade_size))  # AI-AGENT-REF: validate quantity
            price = _finite_pos("price", market_data.get('prices', {}).get(symbol, 100.0))  # AI-AGENT-REF: validate price
            trade_value = trade_size_abs * price

            # Calculate individual cost components
            spread_cost = self.calculate_spread_cost(symbol, trade_size, market_data)
            commission = self.calculate_commission(symbol, trade_size, trade_value)

            temp_impact, perm_impact = self.calculate_market_impact(symbol, trade_size, market_data)
            market_impact = temp_impact + perm_impact

            opportunity_cost = self.calculate_opportunity_cost(symbol, expected_delay, expected_return, trade_value)
            borrowing_cost = self.calculate_borrowing_cost(symbol, trade_size, trade_value, holding_period_days)

            # Adjust costs based on trade type
            if trade_type == TradeType.LIMIT_ORDER:
                # Limit orders have lower spread cost but higher opportunity cost
                spread_cost *= 0.5
                opportunity_cost *= 1.5
            elif trade_type == TradeType.MARKET_ORDER:
                # Market orders have full spread cost but lower opportunity cost
                opportunity_cost *= 0.5

            total_cost = spread_cost + commission + market_impact + opportunity_cost + borrowing_cost
            cost_per_share = total_cost / trade_size_abs
            cost_percentage = total_cost / max(trade_value, 1)

            result = TransactionCostBreakdown(
                spread_cost=spread_cost,
                commission=commission,
                market_impact=market_impact,
                opportunity_cost=opportunity_cost,
                borrowing_cost=borrowing_cost,
                total_cost=total_cost,
                cost_per_share=cost_per_share,
                cost_percentage=cost_percentage
            )

            _log.info(
                f"Transaction cost for {symbol}: ${total_cost:.4f} ("  # AI-AGENT-REF: info log
                f"{cost_percentage:.4f}% of trade value)"
            )

            _log.debug(
                "TX_COSTS_COMPUTED",
                extra={
                    "symbol": symbol,
                    "qty": trade_size_abs,
                    "price": price,
                    "fee_rate": None,
                    "slippage_bps": None,
                    "atr_used": None,
                    "result_keys": list(result.__dict__.keys()),
                },
            )  # AI-AGENT-REF: debug log

            return result

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.error(
                "TX_COST_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            raise

    def validate_trade_profitability(self,
                                   symbol: str,
                                   trade_size: float,
                                   expected_profit: float,
                                   market_data: dict[str, Any],
                                   trade_type: TradeType = TradeType.LIMIT_ORDER,
                                   confidence_level: float = 0.8) -> ProfitabilityAnalysis:
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
            # Calculate transaction costs
            cost_breakdown = self.calculate_total_transaction_cost(
                symbol, trade_size, trade_type, market_data
            )

            transaction_cost = cost_breakdown.total_cost

            # Calculate net profit and margins
            net_expected_profit = expected_profit - transaction_cost
            profit_margin = net_expected_profit / max(abs(expected_profit), 0.01) if expected_profit != 0 else -1
            cost_ratio = transaction_cost / max(abs(expected_profit), 0.01) if expected_profit != 0 else float('inf')

            # Calculate required safety margin
            required_profit = transaction_cost * self.safety_margin_multiplier
            safety_margin = (expected_profit - required_profit) / max(required_profit, 0.01)

            # Determine if trade is profitable
            is_profitable = (
                expected_profit > required_profit and  # Meets safety margin
                net_expected_profit > 0 and           # Positive net profit
                cost_ratio < 0.5 and                  # Costs < 50% of expected profit
                confidence_level >= 0.6               # Sufficient confidence
            )

            _log.info(f"Profitability analysis for {symbol}: "
                       f"profit=${expected_profit:.4f}, cost=${transaction_cost:.4f}, "
                       f"net=${net_expected_profit:.4f}, profitable={is_profitable}")

            return ProfitabilityAnalysis(
                expected_profit=expected_profit,
                transaction_cost=transaction_cost,
                net_expected_profit=net_expected_profit,
                profit_margin=profit_margin,
                cost_ratio=cost_ratio,
                safety_margin=safety_margin,
                is_profitable=is_profitable,
                confidence_level=confidence_level
            )

        except ValueError as e:
            _log.warning(
                "PROFITABILITY_VALIDATION_VALUE_ERROR",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )  # AI-AGENT-REF: structured logging
            raise ValueError(f"Invalid trade parameters: {str(e)}")
        except KeyError as e:
            _log.warning(
                "PROFITABILITY_VALIDATION_KEY_ERROR",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )  # AI-AGENT-REF: structured logging
            raise KeyError(f"Required market data missing: {str(e)}")
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.error(
                "PROFITABILITY_VALIDATION_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            return ProfitabilityAnalysis(
                expected_profit=expected_profit,
                transaction_cost=abs(expected_profit) * 0.5,  # Assume high cost
                net_expected_profit=expected_profit * 0.5,
                profit_margin=-0.5,
                cost_ratio=2.0,
                safety_margin=-1.0,
                is_profitable=False,
                confidence_level=confidence_level
            )

    def _estimate_spread_percentage(self, symbol: str, market_data: dict[str, Any]) -> float:
        """Estimate bid-ask spread percentage based on market data."""
        try:
            # Get liquidity tier
            liquidity_tier = self._classify_liquidity(symbol, market_data)

            # Spread estimates by liquidity tier
            spread_estimates = {
                LiquidityTier.HIGH_LIQUIDITY: 0.0005,    # 5 bps
                LiquidityTier.MEDIUM_LIQUIDITY: 0.002,   # 20 bps
                LiquidityTier.LOW_LIQUIDITY: 0.005,      # 50 bps
                LiquidityTier.ILLIQUID: 0.01             # 100 bps
            }

            return spread_estimates.get(liquidity_tier, 0.005)

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.warning(
                "SPREAD_PERCENT_ESTIMATE_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            return 0.005  # 50 bps default

    def _classify_liquidity(self, symbol: str, market_data: dict[str, Any]) -> LiquidityTier:
        """Classify symbol liquidity based on volume and market cap."""
        try:
            volumes = market_data.get('volumes', {})
            avg_volume = volumes.get(symbol, 100000)

            # Simple classification based on volume
            if avg_volume >= 1000000:
                return LiquidityTier.HIGH_LIQUIDITY
            elif avg_volume >= 100000:
                return LiquidityTier.MEDIUM_LIQUIDITY
            elif avg_volume >= 10000:
                return LiquidityTier.LOW_LIQUIDITY
            else:
                return LiquidityTier.ILLIQUID

        except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:  # AI-AGENT-REF: narrow exception
            _log.warning(
                "LIQUIDITY_CLASSIFICATION_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e), "symbol": symbol},
            )
            return LiquidityTier.MEDIUM_LIQUIDITY


# AI-AGENT-REF: Transaction cost analysis for portfolio optimization
def create_transaction_cost_calculator(config: dict[str, Any] | None = None) -> TransactionCostCalculator:
    """
    Factory function to create transaction cost calculator with configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured TransactionCostCalculator instance
    """
    if config is None:
        config = {}

    return TransactionCostCalculator(
        commission_rate=config.get('commission_rate', 0.0001),
        min_commission=config.get('min_commission', 0.0),
        max_commission=config.get('max_commission', 1.0),
        safety_margin_multiplier=config.get('safety_margin_multiplier', 2.0)
    )
