"""
Portfolio-Level Optimization and Decision Engine.

This module implements portfolio-first trading decisions that dramatically reduce churn
by evaluating trades at the portfolio level rather than individual signal level.
Integrates Kelly Criterion, correlation analysis, and tax-aware rebalancing.
"""

import logging
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Import existing sophisticated infrastructure
try:
    from ai_trading.risk.kelly import KellyCriterion, KellyCalculator
    from ai_trading.risk.adaptive_sizing import AdaptivePositionSizer, MarketRegime
    from ai_trading.core.constants import RISK_PARAMETERS
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced features not available, using fallback implementations")
    ENHANCED_FEATURES_AVAILABLE = False


class PortfolioDecision(Enum):
    """Portfolio-level decision outcomes."""
    APPROVE = "approve"           # Trade improves portfolio metrics significantly
    REJECT = "reject"             # Trade would harm portfolio performance
    DEFER = "defer"              # Trade should wait for better conditions
    BATCH = "batch"              # Trade should be batched with others


@dataclass
class PortfolioMetrics:
    """Current portfolio efficiency metrics."""
    kelly_efficiency: float      # Portfolio-level Kelly efficiency (0-1)
    diversification_ratio: float # Diversification benefit vs concentration
    correlation_penalty: float   # Penalty for high correlations
    transaction_cost_ratio: float # Transaction costs as % of expected return
    tax_efficiency: float        # Tax efficiency score (0-1)
    rebalance_drift: float      # Drift from target allocations
    
    def __post_init__(self):
        """Validate metrics are in expected ranges."""
        self.kelly_efficiency = max(0.0, min(1.0, self.kelly_efficiency))
        self.diversification_ratio = max(0.0, min(2.0, self.diversification_ratio))
        self.tax_efficiency = max(0.0, min(1.0, self.tax_efficiency))


@dataclass
class TradeImpactAnalysis:
    """Analysis of how a trade would impact portfolio."""
    expected_return_change: float    # Change in portfolio expected return
    risk_change: float              # Change in portfolio risk
    kelly_efficiency_change: float  # Change in Kelly efficiency
    correlation_impact: float       # Impact on portfolio correlations
    transaction_cost: float         # Total transaction cost
    tax_impact: float              # Tax implications
    net_benefit: float             # Net benefit after all costs
    confidence: float              # Confidence in the analysis (0-1)


class PortfolioOptimizer:
    """
    Portfolio-Level Decision Engine for Churn Reduction.
    
    Implements sophisticated portfolio optimization that evaluates trades based on
    their impact on overall portfolio performance rather than individual signals.
    """
    
    def __init__(self, 
                 improvement_threshold: float = 0.02,  # 2% minimum improvement
                 max_correlation_penalty: float = 0.15,
                 rebalance_drift_threshold: float = 0.05):
        """
        Initialize portfolio optimizer.
        
        Args:
            improvement_threshold: Minimum portfolio improvement required (2%+)
            max_correlation_penalty: Maximum correlation penalty allowed
            rebalance_drift_threshold: Drift threshold for rebalancing trigger
        """
        self.improvement_threshold = improvement_threshold
        self.max_correlation_penalty = max_correlation_penalty
        self.rebalance_drift_threshold = rebalance_drift_threshold
        
        # Initialize enhanced components if available
        if ENHANCED_FEATURES_AVAILABLE:
            self.kelly_calculator = KellyCriterion()
            self.adaptive_sizer = AdaptivePositionSizer()
            # Use late import to avoid circular import
            try:
                from ai_trading.rebalancer import TaxAwareRebalancer
                self.tax_rebalancer = TaxAwareRebalancer()
            except ImportError:
                self.tax_rebalancer = None
            logger.info("PortfolioOptimizer initialized with enhanced features")
        else:
            # Fallback implementations
            self.kelly_calculator = None
            self.adaptive_sizer = None
            self.tax_rebalancer = None
            logger.warning("PortfolioOptimizer initialized with fallback features")
        
        # Portfolio state tracking
        self.current_metrics: Optional[PortfolioMetrics] = None
        self.last_rebalance: Optional[datetime] = None
        
        logger.info(f"PortfolioOptimizer initialized with improvement_threshold={improvement_threshold:.1%}")

    def calculate_portfolio_kelly_efficiency(self, 
                                          positions: Dict[str, float],
                                          returns_data: Dict[str, List[float]],
                                          current_prices: Dict[str, float]) -> float:
        """
        Calculate portfolio-level Kelly efficiency score.
        
        Args:
            positions: Current position sizes {symbol: size}
            returns_data: Historical returns {symbol: [returns]}
            current_prices: Current market prices {symbol: price}
            
        Returns:
            Kelly efficiency score (0-1), higher is better
        """
        try:
            if not positions or not returns_data:
                return 0.0
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(positions, returns_data, current_prices)
            
            if len(portfolio_returns) < 20:  # Minimum sample size
                logger.warning("Insufficient data for Kelly efficiency calculation")
                return 0.0
            
            # Calculate Kelly metrics for the portfolio
            positive_returns = [r for r in portfolio_returns if r > 0]
            negative_returns = [abs(r) for r in portfolio_returns if r < 0]
            
            if not positive_returns or not negative_returns:
                return 0.0
            
            win_rate = len(positive_returns) / len(portfolio_returns)
            avg_win = statistics.mean(positive_returns)
            avg_loss = statistics.mean(negative_returns)
            
            if ENHANCED_FEATURES_AVAILABLE and self.kelly_calculator:
                optimal_fraction = self.kelly_calculator.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            else:
                # Fallback Kelly calculation
                if avg_loss == 0:
                    return 0.0
                b = avg_win / avg_loss
                optimal_fraction = max(0, min(0.25, (b * win_rate - (1 - win_rate)) / b))
            
            # Calculate current leverage vs optimal
            current_leverage = sum(abs(pos) for pos in positions.values())
            efficiency = min(1.0, optimal_fraction / max(0.01, current_leverage))
            
            logger.debug(f"Portfolio Kelly efficiency: {efficiency:.3f} (optimal_fraction={optimal_fraction:.3f})")
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Kelly efficiency: {e}")
            return 0.0

    def _calculate_portfolio_returns(self, 
                                   positions: Dict[str, float],
                                   returns_data: Dict[str, List[float]],
                                   current_prices: Dict[str, float]) -> List[float]:
        """Calculate historical portfolio returns based on current positions."""
        if not positions or not returns_data:
            return []
        
        # Get minimum length across all symbols
        min_length = min(len(returns_data[symbol]) for symbol in positions.keys() 
                        if symbol in returns_data)
        
        if min_length == 0:
            return []
        
        # Calculate portfolio value weights
        total_value = sum(abs(positions[symbol]) * current_prices.get(symbol, 1.0) 
                         for symbol in positions.keys())
        
        if total_value == 0:
            return []
        
        portfolio_returns = []
        for i in range(min_length):
            period_return = 0.0
            for symbol, position in positions.items():
                if symbol in returns_data and i < len(returns_data[symbol]):
                    weight = (abs(position) * current_prices.get(symbol, 1.0)) / total_value
                    period_return += weight * returns_data[symbol][i]
            portfolio_returns.append(period_return)
        
        return portfolio_returns

    def calculate_correlation_impact(self, 
                                   new_symbol: str,
                                   current_positions: Dict[str, float],
                                   correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate impact of adding/modifying position on portfolio correlations.
        
        Args:
            new_symbol: Symbol being considered
            current_positions: Current portfolio positions
            correlation_matrix: Correlation matrix between symbols
            
        Returns:
            Correlation penalty (0-1), higher means more correlated
        """
        try:
            if not current_positions or not correlation_matrix:
                return 0.0
            
            if new_symbol not in correlation_matrix:
                return 0.0  # No correlation data available
            
            # Calculate weighted correlation with existing positions
            total_correlation = 0.0
            total_weight = 0.0
            
            for symbol, position in current_positions.items():
                if symbol != new_symbol and symbol in correlation_matrix[new_symbol]:
                    weight = abs(position)
                    correlation = abs(correlation_matrix[new_symbol][symbol])
                    total_correlation += weight * correlation
                    total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            avg_correlation = total_correlation / total_weight
            
            # Convert to penalty score (higher correlation = higher penalty)
            correlation_penalty = min(1.0, avg_correlation)
            
            logger.debug(f"Correlation impact for {new_symbol}: {correlation_penalty:.3f}")
            return correlation_penalty
            
        except Exception as e:
            logger.error(f"Error calculating correlation impact: {e}")
            return 0.0

    def evaluate_trade_impact(self, 
                            symbol: str,
                            proposed_position: float,
                            current_positions: Dict[str, float],
                            market_data: Dict[str, Any]) -> TradeImpactAnalysis:
        """
        Evaluate the impact of a proposed trade on portfolio performance.
        
        Args:
            symbol: Symbol to trade
            proposed_position: Proposed new position size
            current_positions: Current portfolio positions
            market_data: Market data including prices, returns, correlations
            
        Returns:
            Comprehensive trade impact analysis
        """
        try:
            current_position = current_positions.get(symbol, 0.0)
            position_change = proposed_position - current_position
            
            # Extract market data
            current_prices = market_data.get('prices', {})
            returns_data = market_data.get('returns', {})
            correlation_matrix = market_data.get('correlations', {})
            
            # Calculate transaction cost
            transaction_cost = self._estimate_transaction_cost(symbol, abs(position_change), current_prices)
            
            # Calculate correlation impact
            correlation_impact = self.calculate_correlation_impact(symbol, current_positions, correlation_matrix)
            
            # Simulate new portfolio
            new_positions = current_positions.copy()
            new_positions[symbol] = proposed_position
            
            # Calculate Kelly efficiency change
            current_efficiency = self.calculate_portfolio_kelly_efficiency(current_positions, returns_data, current_prices)
            new_efficiency = self.calculate_portfolio_kelly_efficiency(new_positions, returns_data, current_prices)
            kelly_efficiency_change = new_efficiency - current_efficiency
            
            # Estimate expected return change (simplified)
            expected_return_change = self._estimate_return_change(symbol, position_change, market_data)
            
            # Estimate risk change
            risk_change = self._estimate_risk_change(symbol, position_change, current_positions, market_data)
            
            # Calculate tax impact
            tax_impact = self._estimate_tax_impact(symbol, position_change, current_prices)
            
            # Calculate net benefit
            net_benefit = expected_return_change - transaction_cost - (correlation_impact * self.max_correlation_penalty) + tax_impact
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(symbol, market_data)
            
            return TradeImpactAnalysis(
                expected_return_change=expected_return_change,
                risk_change=risk_change,
                kelly_efficiency_change=kelly_efficiency_change,
                correlation_impact=correlation_impact,
                transaction_cost=transaction_cost,
                tax_impact=tax_impact,
                net_benefit=net_benefit,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error evaluating trade impact for {symbol}: {e}")
            return TradeImpactAnalysis(0, 0, 0, 0, 0, 0, -1, 0)  # Reject by default

    def make_portfolio_decision(self, 
                              symbol: str,
                              proposed_position: float,
                              current_positions: Dict[str, float],
                              market_data: Dict[str, Any]) -> Tuple[PortfolioDecision, str]:
        """
        Make portfolio-level decision about whether to execute a trade.
        
        Args:
            symbol: Symbol to trade
            proposed_position: Proposed new position size
            current_positions: Current portfolio positions
            market_data: Market data for analysis
            
        Returns:
            Tuple of (decision, reasoning)
        """
        try:
            # Analyze trade impact
            impact = self.evaluate_trade_impact(symbol, proposed_position, current_positions, market_data)
            
            # Apply decision logic with multiple criteria
            
            # 1. Check if net benefit meets improvement threshold
            if impact.net_benefit < self.improvement_threshold:
                return PortfolioDecision.REJECT, f"Net benefit {impact.net_benefit:.3f} below threshold {self.improvement_threshold:.3f}"
            
            # 2. Check transaction cost vs expected return
            if impact.transaction_cost > abs(impact.expected_return_change) * 0.5:
                return PortfolioDecision.REJECT, f"Transaction cost {impact.transaction_cost:.3f} too high vs expected return {impact.expected_return_change:.3f}"
            
            # 3. Check correlation penalty
            if impact.correlation_impact > self.max_correlation_penalty:
                return PortfolioDecision.DEFER, f"Correlation impact {impact.correlation_impact:.3f} exceeds maximum {self.max_correlation_penalty:.3f}"
            
            # 4. Check Kelly efficiency improvement
            if impact.kelly_efficiency_change < -0.05:  # Don't reduce efficiency by more than 5%
                return PortfolioDecision.REJECT, f"Kelly efficiency would decrease by {abs(impact.kelly_efficiency_change):.3f}"
            
            # 5. Check confidence level
            if impact.confidence < 0.6:
                return PortfolioDecision.DEFER, f"Low confidence {impact.confidence:.3f} in analysis"
            
            # 6. Consider batching for small improvements
            if 0.005 <= impact.net_benefit < self.improvement_threshold:
                return PortfolioDecision.BATCH, f"Small improvement {impact.net_benefit:.3f} suitable for batching"
            
            # Approve trade
            return PortfolioDecision.APPROVE, f"Portfolio improvement: {impact.net_benefit:.3f}, Kelly change: {impact.kelly_efficiency_change:.3f}"
            
        except Exception as e:
            logger.error(f"Error making portfolio decision: {e}")
            return PortfolioDecision.REJECT, f"Analysis error: {str(e)}"

    def should_trigger_rebalance(self, 
                               current_positions: Dict[str, float],
                               target_weights: Dict[str, float],
                               current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if portfolio should be rebalanced based on drift and tax considerations.
        
        Args:
            current_positions: Current position sizes
            target_weights: Target portfolio weights
            current_prices: Current market prices
            
        Returns:
            Tuple of (should_rebalance, reasoning)
        """
        try:
            # Calculate current weights
            total_value = sum(abs(pos) * current_prices.get(symbol, 1.0) 
                            for symbol, pos in current_positions.items())
            
            if total_value == 0:
                return False, "No positions to rebalance"
            
            current_weights = {
                symbol: (abs(pos) * current_prices.get(symbol, 1.0)) / total_value
                for symbol, pos in current_positions.items()
            }
            
            # Calculate drift from target
            max_drift = 0.0
            total_drift = 0.0
            
            for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                drift = abs(current_weight - target_weight)
                max_drift = max(max_drift, drift)
                total_drift += drift
            
            # Check drift thresholds
            if max_drift > self.rebalance_drift_threshold:
                return True, f"Maximum drift {max_drift:.3f} exceeds threshold {self.rebalance_drift_threshold:.3f}"
            
            # Check time-based rebalancing (quarterly for tax efficiency)
            if self.last_rebalance is not None:
                days_since_rebalance = (datetime.now(timezone.utc) - self.last_rebalance).days
                if days_since_rebalance >= 90:  # Quarterly rebalancing
                    return True, f"Quarterly rebalance due ({days_since_rebalance} days since last)"
            
            return False, f"Drift within tolerance (max: {max_drift:.3f}, total: {total_drift:.3f})"
            
        except Exception as e:
            logger.error(f"Error checking rebalance trigger: {e}")
            return False, f"Error: {str(e)}"

    def _estimate_transaction_cost(self, symbol: str, trade_size: float, current_prices: Dict[str, float]) -> float:
        """Estimate transaction cost for a trade."""
        try:
            price = current_prices.get(symbol, 100.0)  # Default price if missing
            trade_value = abs(trade_size) * price
            
            # Simplified cost model: spread + commission + market impact
            spread_cost = trade_value * 0.001  # 10 bps spread
            commission = min(1.0, trade_value * 0.0001)  # Commission with cap
            market_impact = trade_value * 0.0005  # 5 bps market impact
            
            total_cost = spread_cost + commission + market_impact
            return total_cost
            
        except Exception:
            return 0.01  # Default small cost

    def _estimate_return_change(self, symbol: str, position_change: float, market_data: Dict[str, Any]) -> float:
        """Estimate expected return change from position modification."""
        try:
            returns_data = market_data.get('returns', {})
            if symbol not in returns_data or len(returns_data[symbol]) < 10:
                return 0.0
            
            # Use recent average return as expectation
            recent_returns = returns_data[symbol][-10:]  # Last 10 periods
            avg_return = statistics.mean(recent_returns)
            
            # Return change proportional to position change
            return position_change * avg_return
            
        except Exception:
            return 0.0

    def _estimate_risk_change(self, symbol: str, position_change: float, 
                            current_positions: Dict[str, float], market_data: Dict[str, Any]) -> float:
        """Estimate portfolio risk change from position modification."""
        try:
            returns_data = market_data.get('returns', {})
            if symbol not in returns_data:
                return 0.0
            
            # Simplified risk estimate based on volatility
            symbol_returns = returns_data[symbol]
            if len(symbol_returns) < 10:
                return 0.0
            
            volatility = statistics.stdev(symbol_returns)
            risk_change = abs(position_change) * volatility
            
            return risk_change
            
        except Exception:
            return 0.0

    def _estimate_tax_impact(self, symbol: str, position_change: float, current_prices: Dict[str, float]) -> float:
        """Estimate tax impact of trade."""
        try:
            if ENHANCED_FEATURES_AVAILABLE and self.tax_rebalancer:
                # Use sophisticated tax calculation if available
                return 0.0  # Placeholder for actual tax calculation
            else:
                # Simplified tax impact
                trade_value = abs(position_change) * current_prices.get(symbol, 100.0)
                return -trade_value * 0.15  # Assume 15% tax drag
                
        except Exception:
            return 0.0

    def _calculate_confidence(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate confidence in analysis based on data quality."""
        try:
            confidence = 1.0
            
            # Penalize missing data
            if symbol not in market_data.get('prices', {}):
                confidence *= 0.5
            
            returns_data = market_data.get('returns', {})
            if symbol not in returns_data:
                confidence *= 0.3
            elif len(returns_data[symbol]) < 20:
                confidence *= 0.7
            
            correlations = market_data.get('correlations', {})
            if symbol not in correlations:
                confidence *= 0.8
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Medium confidence on error


# AI-AGENT-REF: Portfolio-level optimization for churn reduction
def create_portfolio_optimizer(config: Optional[Dict[str, Any]] = None) -> PortfolioOptimizer:
    """
    Factory function to create portfolio optimizer with configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured PortfolioOptimizer instance
    """
    if config is None:
        config = {}
    
    return PortfolioOptimizer(
        improvement_threshold=config.get('improvement_threshold', 0.02),
        max_correlation_penalty=config.get('max_correlation_penalty', 0.15),
        rebalance_drift_threshold=config.get('rebalance_drift_threshold', 0.05)
    )