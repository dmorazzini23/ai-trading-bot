"""
Advanced position sizing engine for production trading.

Implements comprehensive position sizing algorithms including ATR-based sizing,
volatility-adjusted Kelly criterion, and dynamic risk-based position sizing
for institutional-grade trading operations.
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

# Use the centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..core.enums import RiskLevel
from ..core.constants import RISK_PARAMETERS, KELLY_PARAMETERS
from .kelly import KellyCriterion


class ATRPositionSizer:
    """
    ATR-based position sizing for volatility-adjusted risk management.
    
    Uses Average True Range (ATR) to determine position sizes that
    maintain consistent risk across different volatility regimes.
    """
    
    def __init__(self, risk_per_trade: float = None):
        """Initialize ATR position sizer."""
        # AI-AGENT-REF: ATR-based dynamic position sizing
        self.risk_per_trade = risk_per_trade or RISK_PARAMETERS["MAX_PORTFOLIO_RISK"]
        self.atr_multiplier = RISK_PARAMETERS["STOP_LOSS_MULTIPLIER"]
        
        logger.info(f"ATRPositionSizer initialized with risk_per_trade={self.risk_per_trade}, "
                   f"atr_multiplier={self.atr_multiplier}")
    
    def calculate_position_size(self, account_equity: float, entry_price: float, 
                              atr_value: float, stop_distance_multiplier: float = None) -> int:
        """
        Calculate position size based on ATR and account equity.
        
        Args:
            account_equity: Total account equity
            entry_price: Planned entry price
            atr_value: Current ATR value for the symbol
            stop_distance_multiplier: Multiplier for ATR to determine stop distance
            
        Returns:
            Recommended position size in shares
        """
        try:
            if account_equity <= 0 or entry_price <= 0 or atr_value <= 0:
                logger.warning(f"Invalid inputs: equity={account_equity}, price={entry_price}, atr={atr_value}")
                return 0
            
            # Use provided multiplier or default
            multiplier = stop_distance_multiplier or self.atr_multiplier
            
            # Calculate risk amount in dollars
            risk_amount = account_equity * self.risk_per_trade
            
            # Calculate stop distance in dollars
            stop_distance = atr_value * multiplier
            
            # Calculate position size
            position_size = int(risk_amount / stop_distance)
            
            # Ensure minimum position size
            position_size = max(1, position_size)
            
            logger.debug(f"ATR position sizing: equity=${account_equity:,.2f}, "
                        f"risk_amount=${risk_amount:,.2f}, stop_distance=${stop_distance:.2f}, "
                        f"position_size={position_size}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating ATR position size: {e}")
            return 0
    
    def calculate_stop_levels(self, entry_price: float, atr_value: float, 
                            side: str = "long") -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels based on ATR.
        
        Args:
            entry_price: Entry price for the position
            atr_value: Current ATR value
            side: Position side ("long" or "short")
            
        Returns:
            Dictionary with stop_loss and take_profit levels
        """
        try:
            if entry_price <= 0 or atr_value <= 0:
                return {"stop_loss": 0.0, "take_profit": 0.0}
            
            stop_distance = atr_value * self.atr_multiplier
            profit_distance = atr_value * RISK_PARAMETERS["TAKE_PROFIT_MULTIPLIER"]
            
            if side.lower() == "long":
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + profit_distance
            else:  # short
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - profit_distance
            
            return {
                "stop_loss": max(0.01, stop_loss),  # Ensure positive prices
                "take_profit": max(0.01, take_profit)
            }
            
        except Exception as e:
            logger.error(f"Error calculating stop levels: {e}")
            return {"stop_loss": 0.0, "take_profit": 0.0}


class VolatilityPositionSizer:
    """
    Volatility-based position sizing using historical volatility measures.
    
    Adjusts position sizes based on recent volatility to maintain
    consistent risk across different market conditions.
    """
    
    def __init__(self, target_volatility: float = 0.15, lookback_days: int = 20):
        """Initialize volatility position sizer."""
        # AI-AGENT-REF: Volatility-based position sizing
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        
        logger.info(f"VolatilityPositionSizer initialized with target_volatility={target_volatility}, "
                   f"lookback_days={lookback_days}")
    
    def calculate_volatility_multiplier(self, returns: List[float]) -> float:
        """
        Calculate volatility multiplier for position sizing.
        
        Args:
            returns: List of recent returns for volatility calculation
            
        Returns:
            Volatility multiplier (1.0 = normal, <1.0 = high vol, >1.0 = low vol)
        """
        try:
            if len(returns) < 10:
                logger.warning(f"Insufficient data for volatility calculation: {len(returns)} returns")
                return 1.0
            
            # Calculate realized volatility (annualized)
            returns_std = statistics.stdev(returns)
            realized_volatility = returns_std * math.sqrt(252)  # Annualize
            
            # Calculate volatility multiplier
            if realized_volatility > 0:
                volatility_multiplier = self.target_volatility / realized_volatility
                # Cap the multiplier to reasonable bounds
                volatility_multiplier = max(0.2, min(2.0, volatility_multiplier))
            else:
                volatility_multiplier = 1.0
            
            logger.debug(f"Volatility calculation: realized={realized_volatility:.3f}, "
                        f"target={self.target_volatility:.3f}, multiplier={volatility_multiplier:.3f}")
            
            return volatility_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating volatility multiplier: {e}")
            return 1.0
    
    def calculate_position_size(self, base_position_size: int, returns: List[float]) -> int:
        """
        Calculate volatility-adjusted position size.
        
        Args:
            base_position_size: Base position size before volatility adjustment
            returns: Recent returns for volatility calculation
            
        Returns:
            Volatility-adjusted position size
        """
        try:
            volatility_multiplier = self.calculate_volatility_multiplier(returns)
            adjusted_size = int(base_position_size * volatility_multiplier)
            
            logger.debug(f"Volatility position sizing: base={base_position_size}, "
                        f"multiplier={volatility_multiplier:.3f}, adjusted={adjusted_size}")
            
            return max(1, adjusted_size)
            
        except Exception as e:
            logger.error(f"Error calculating volatility-adjusted position size: {e}")
            return base_position_size


class DynamicPositionSizer:
    """
    Dynamic position sizing engine that combines multiple sizing methods.
    
    Integrates ATR-based sizing, volatility adjustments, Kelly criterion,
    and account equity considerations for optimal position sizing.
    """
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        """Initialize dynamic position sizer."""
        # AI-AGENT-REF: Comprehensive dynamic position sizing system
        self.risk_level = risk_level
        self.atr_sizer = ATRPositionSizer()
        self.volatility_sizer = VolatilityPositionSizer()
        self.kelly_criterion = KellyCriterion()
        
        # Configuration
        self.max_position_pct = risk_level.max_position_size
        self.base_risk_per_trade = RISK_PARAMETERS["MAX_PORTFOLIO_RISK"]
        
        logger.info(f"DynamicPositionSizer initialized with risk_level={risk_level}")
    
    def calculate_optimal_position(self, symbol: str, account_equity: float, 
                                 entry_price: float, market_data: Dict, 
                                 historical_data: Dict) -> Dict[str, Any]:
        """
        Calculate optimal position size using multiple methods.
        
        Args:
            symbol: Trading symbol
            account_equity: Current account equity
            entry_price: Planned entry price
            market_data: Current market data including ATR
            historical_data: Historical price and return data
            
        Returns:
            Dictionary with position sizing recommendations
        """
        try:
            result = {
                "symbol": symbol,
                "recommended_size": 0,
                "sizing_methods": {},
                "risk_metrics": {},
                "warnings": []
            }
            
            # Extract required data
            atr_value = market_data.get("atr", 0.0)
            returns = historical_data.get("returns", [])
            trade_history = historical_data.get("trade_history", [])
            
            if atr_value <= 0:
                result["warnings"].append("Invalid ATR value, using conservative sizing")
                atr_value = entry_price * 0.02  # 2% fallback
            
            # Method 1: ATR-based sizing
            atr_size = self.atr_sizer.calculate_position_size(
                account_equity, entry_price, atr_value
            )
            result["sizing_methods"]["atr_based"] = atr_size
            
            # Method 2: Volatility-adjusted sizing
            if returns:
                vol_adjusted_size = self.volatility_sizer.calculate_position_size(atr_size, returns)
                result["sizing_methods"]["volatility_adjusted"] = vol_adjusted_size
            else:
                vol_adjusted_size = atr_size
                result["warnings"].append("No return data for volatility adjustment")
            
            # Method 3: Kelly-based sizing
            kelly_size = atr_size  # Default fallback
            if trade_history and len(trade_history) >= self.kelly_criterion.min_sample_size:
                kelly_fraction, kelly_stats = self.kelly_criterion.calculate_from_returns(
                    [trade.get("return", 0.0) for trade in trade_history]
                )
                if kelly_fraction > 0:
                    kelly_notional = account_equity * kelly_fraction
                    kelly_size = int(kelly_notional / entry_price)
                    result["sizing_methods"]["kelly_based"] = kelly_size
                    result["risk_metrics"]["kelly_fraction"] = kelly_fraction
                    result["risk_metrics"]["kelly_stats"] = kelly_stats
            
            # Method 4: Portfolio concentration limits
            max_notional = account_equity * self.max_position_pct
            max_shares = int(max_notional / entry_price)
            result["sizing_methods"]["concentration_limit"] = max_shares
            
            # Calculate final recommendation (take minimum for safety)
            recommended_sizes = [
                size for size in [atr_size, vol_adjusted_size, kelly_size, max_shares]
                if size > 0
            ]
            
            if recommended_sizes:
                result["recommended_size"] = min(recommended_sizes)
            else:
                result["recommended_size"] = 0
                result["warnings"].append("No valid position size calculated")
            
            # Calculate risk metrics
            if result["recommended_size"] > 0:
                notional_value = result["recommended_size"] * entry_price
                position_pct = notional_value / account_equity
                
                result["risk_metrics"]["notional_value"] = notional_value
                result["risk_metrics"]["position_percentage"] = position_pct
                result["risk_metrics"]["estimated_risk"] = self._estimate_position_risk(
                    notional_value, atr_value
                )
            
            # Generate warnings for large positions
            if result["recommended_size"] * entry_price > account_equity * 0.05:
                result["warnings"].append("Large position size relative to account")
            
            logger.info(f"Position sizing for {symbol}: recommended={result['recommended_size']}, "
                       f"methods={result['sizing_methods']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating optimal position for {symbol}: {e}")
            return {
                "symbol": symbol,
                "recommended_size": 0,
                "sizing_methods": {},
                "risk_metrics": {},
                "warnings": [f"Position sizing error: {e}"]
            }
    
    def calculate_scaling_orders(self, base_position: int, entry_price: float, 
                               scaling_levels: int = 3) -> List[Dict]:
        """
        Calculate scaled order entries for large positions.
        
        Args:
            base_position: Total desired position size
            entry_price: Target entry price
            scaling_levels: Number of scaling levels
            
        Returns:
            List of scaled order dictionaries
        """
        try:
            if base_position <= 0 or scaling_levels <= 0:
                return []
            
            scaled_orders = []
            remaining_size = base_position
            
            for level in range(scaling_levels):
                # Use Fibonacci-like scaling: larger orders first
                if level == 0:
                    order_size = min(remaining_size, int(base_position * 0.5))
                elif level == 1:
                    order_size = min(remaining_size, int(base_position * 0.3))
                else:
                    order_size = remaining_size
                
                if order_size > 0:
                    # Add small price improvement for subsequent orders
                    price_adjustment = entry_price * 0.001 * level  # 0.1% per level
                    
                    scaled_orders.append({
                        "level": level + 1,
                        "size": order_size,
                        "price": entry_price - price_adjustment,
                        "percentage": order_size / base_position * 100
                    })
                    
                    remaining_size -= order_size
                
                if remaining_size <= 0:
                    break
            
            logger.debug(f"Scaled orders for position {base_position}: {len(scaled_orders)} levels")
            return scaled_orders
            
        except Exception as e:
            logger.error(f"Error calculating scaling orders: {e}")
            return []
    
    def _estimate_position_risk(self, notional_value: float, atr_value: float) -> float:
        """Estimate the risk of a position based on ATR."""
        try:
            if notional_value <= 0 or atr_value <= 0:
                return 0.0
            
            # Estimate risk as ATR percentage of notional value
            risk_estimate = (atr_value * self.atr_sizer.atr_multiplier) / (notional_value / 100)
            return min(1.0, risk_estimate)  # Cap at 100%
            
        except Exception:
            return 0.0


class PortfolioPositionManager:
    """
    Portfolio-level position management and coordination.
    
    Manages position sizing across the entire portfolio to maintain
    risk limits, diversification, and optimal capital allocation.
    """
    
    def __init__(self, max_portfolio_risk: float = None):
        """Initialize portfolio position manager."""
        # AI-AGENT-REF: Portfolio-level position management
        self.max_portfolio_risk = max_portfolio_risk or RISK_PARAMETERS["MAX_PORTFOLIO_RISK"]
        self.dynamic_sizer = DynamicPositionSizer()
        
        # Portfolio state
        self.current_positions = {}
        self.position_correlations = {}
        self.total_risk_exposure = 0.0
        
        logger.info(f"PortfolioPositionManager initialized with max_portfolio_risk={self.max_portfolio_risk}")
    
    def assess_new_position(self, symbol: str, proposed_size: int, 
                          entry_price: float, account_equity: float) -> Dict[str, Any]:
        """
        Assess impact of a new position on portfolio risk.
        
        Args:
            symbol: Trading symbol
            proposed_size: Proposed position size
            entry_price: Entry price
            account_equity: Current account equity
            
        Returns:
            Assessment dictionary with approval and adjustments
        """
        try:
            assessment = {
                "symbol": symbol,
                "approved": False,
                "adjusted_size": 0,
                "risk_impact": 0.0,
                "warnings": [],
                "recommendations": []
            }
            
            # Calculate position value and risk
            notional_value = proposed_size * entry_price
            position_pct = notional_value / account_equity if account_equity > 0 else 0
            
            # Check individual position limits
            if position_pct > self.dynamic_sizer.max_position_pct:
                max_notional = account_equity * self.dynamic_sizer.max_position_pct
                adjusted_size = int(max_notional / entry_price)
                assessment["adjusted_size"] = adjusted_size
                assessment["warnings"].append(f"Position size adjusted from {proposed_size} to {adjusted_size}")
            else:
                assessment["adjusted_size"] = proposed_size
            
            # Estimate portfolio risk impact
            estimated_risk_addition = position_pct * 0.5  # Conservative estimate
            new_total_risk = self.total_risk_exposure + estimated_risk_addition
            
            assessment["risk_impact"] = estimated_risk_addition
            
            # Check portfolio risk limits
            if new_total_risk > self.max_portfolio_risk:
                risk_reduction_needed = new_total_risk - self.max_portfolio_risk
                size_reduction_pct = risk_reduction_needed / estimated_risk_addition
                final_size = int(assessment["adjusted_size"] * (1 - size_reduction_pct))
                
                assessment["adjusted_size"] = max(0, final_size)
                assessment["warnings"].append(f"Portfolio risk limit requires size reduction to {final_size}")
            
            # Final approval check
            assessment["approved"] = (
                assessment["adjusted_size"] > 0 and
                len(assessment["warnings"]) == 0
            )
            
            if assessment["adjusted_size"] != proposed_size:
                assessment["recommendations"].append("Consider splitting order across multiple sessions")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing new position for {symbol}: {e}")
            return {
                "symbol": symbol,
                "approved": False,
                "adjusted_size": 0,
                "risk_impact": 0.0,
                "warnings": [f"Assessment error: {e}"],
                "recommendations": ["Manual review required"]
            }
    
    def update_position(self, symbol: str, size: int, entry_price: float):
        """Update portfolio position tracking."""
        try:
            if size > 0:
                self.current_positions[symbol] = {
                    "size": size,
                    "entry_price": entry_price,
                    "notional_value": size * entry_price,
                    "timestamp": datetime.now()
                }
            else:
                self.current_positions.pop(symbol, None)
            
            # Recalculate total risk exposure
            self._recalculate_risk_exposure()
            
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio position summary."""
        try:
            total_notional = sum(pos["notional_value"] for pos in self.current_positions.values())
            position_count = len(self.current_positions)
            
            return {
                "position_count": position_count,
                "total_notional_value": total_notional,
                "total_risk_exposure": self.total_risk_exposure,
                "positions": dict(self.current_positions),
                "largest_position": max(
                    (pos["notional_value"] for pos in self.current_positions.values()),
                    default=0
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {"error": str(e)}
    
    def _recalculate_risk_exposure(self):
        """Recalculate total portfolio risk exposure."""
        try:
            # Simple risk calculation - in production would use more sophisticated methods
            self.total_risk_exposure = sum(
                pos["notional_value"] * 0.02  # Assume 2% risk per position
                for pos in self.current_positions.values()
            )
            
        except Exception as e:
            logger.error(f"Error recalculating risk exposure: {e}")
            self.total_risk_exposure = 0.0