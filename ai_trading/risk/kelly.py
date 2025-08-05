"""
Kelly Criterion implementation for optimal position sizing.

Provides Kelly Criterion calculation, position sizing optimization,
and risk-adjusted capital allocation for institutional trading.
"""

import math
import statistics
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Import centralized configuration
try:
    from config import TradingConfig
    _DEFAULT_CONFIG = TradingConfig.from_env()
except ImportError:
    # Fallback if config import fails
    logger.warning("Could not import centralized config, using defaults")
    class _DefaultConfig:
        min_sample_size = 20
        kelly_fraction_max = 0.25
        confidence_level = 0.90
        lookback_periods = 252
        rebalance_frequency = 21
    _DEFAULT_CONFIG = _DefaultConfig()


class KellyCriterion:
    """
    Kelly Criterion calculator for optimal position sizing.
    
    Implements the Kelly formula: f = (bp - q) / b
    where:
    - f = fraction of capital to bet
    - b = odds received (win/loss ratio)
    - p = probability of winning
    - q = probability of losing (1-p)
    """
    
    def __init__(self, config: Optional[TradingConfig] = None, min_sample_size: Optional[int] = None, max_fraction: Optional[float] = None, **kwargs):
        """Initialize Kelly Criterion calculator with centralized configuration.
        
        Args:
            config: TradingConfig instance (optional, uses default if not provided)
            min_sample_size: Minimum sample size for calculations (for backward compatibility)
            max_fraction: Maximum Kelly fraction allowed (for backward compatibility)
            **kwargs: Additional parameters for backward compatibility
        """
        # Use provided config or default
        self.config = config or _DEFAULT_CONFIG
        
        # Support backward compatibility: use passed parameters if provided, otherwise use config
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.config.min_sample_size
        self.max_fraction = max_fraction if max_fraction is not None else self.config.kelly_fraction_max
        # For backward compatibility, if no confidence_level in kwargs but we have explicit params, use 0.95 (old default)
        if 'confidence_level' in kwargs:
            self.confidence_level = kwargs['confidence_level']
        elif min_sample_size is not None or max_fraction is not None:
            # If using legacy API with explicit params, use legacy default
            self.confidence_level = 0.95
        else:
            # If using new API, use config value
            self.confidence_level = self.config.confidence_level
        
        logger.info(f"KellyCriterion initialized with min_sample_size={self.min_sample_size}, "
                   f"max_fraction={self.max_fraction}, confidence_level={self.confidence_level}")
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate the optimal Kelly fraction for position sizing.
        
        Args:
            win_rate: Probability of winning trades (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            
        Returns:
            Optimal fraction of capital to risk (0-max_fraction)
        """
        try:
            if win_rate <= 0 or win_rate >= 1:
                logger.warning(f"Invalid win_rate: {win_rate}. Must be between 0 and 1.")
                return 0.0
            
            if avg_win <= 0 or avg_loss <= 0:
                logger.warning(f"Invalid win/loss values: avg_win={avg_win}, avg_loss={avg_loss}")
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate            # Probability of winning
            q = 1 - win_rate        # Probability of losing
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety constraints
            kelly_fraction = max(0.0, kelly_fraction)  # Never go negative
            kelly_fraction = min(kelly_fraction, self.max_fraction)  # Cap maximum
            
            logger.debug(f"Kelly calculation: win_rate={win_rate:.3f}, avg_win={avg_win:.3f}, "
                        f"avg_loss={avg_loss:.3f}, kelly_fraction={kelly_fraction:.3f}")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0
    
    def calculate_from_returns(self, returns: List[float]) -> Tuple[float, Dict]:
        """
        Calculate Kelly fraction from a series of trade returns.
        
        Args:
            returns: List of trade returns (positive for wins, negative for losses)
            
        Returns:
            Tuple of (kelly_fraction, statistics_dict)
        """
        try:
            if len(returns) < self.min_sample_size:
                logger.warning(f"Insufficient sample size: {len(returns)} < {self.min_sample_size}")
                return 0.0, {"error": "Insufficient sample size"}
            
            # Separate wins and losses
            wins = [r for r in returns if r > 0]
            losses = [abs(r) for r in returns if r < 0]
            
            if not wins or not losses:
                logger.warning("No wins or losses found in returns")
                return 0.0, {"error": "No wins or losses"}
            
            # Calculate statistics
            win_rate = len(wins) / len(returns)
            avg_win = statistics.mean(wins)
            avg_loss = statistics.mean(losses)
            
            kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            
            # Additional statistics
            stats = {
                "total_trades": len(returns),
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": (sum(wins) / sum(losses)) if sum(losses) > 0 else float('inf'),
                "kelly_fraction": kelly_fraction,
                "max_kelly_fraction": self.max_fraction,
                "expectancy": win_rate * avg_win - (1 - win_rate) * avg_loss,
            }
            
            return kelly_fraction, stats
            
        except Exception as e:
            logger.error(f"Error calculating Kelly from returns: {e}")
            return 0.0, {"error": str(e)}
    
    def fractional_kelly(self, kelly_fraction: float, fraction: float = 0.25) -> float:
        """
        Apply fractional Kelly to reduce risk.
        
        Args:
            kelly_fraction: Full Kelly fraction
            fraction: Fraction of Kelly to use (default 25%)
            
        Returns:
            Fractional Kelly position size
        """
        return kelly_fraction * fraction
    
    def kelly_with_confidence(self, returns: List[float], confidence: float = None) -> Tuple[float, float]:
        """
        Calculate Kelly fraction with confidence intervals.
        
        Args:
            returns: List of trade returns
            confidence: Confidence level (default from config)
            
        Returns:
            Tuple of (kelly_fraction, confidence_interval_width)
        """
        try:
            confidence = confidence or self.confidence_level
            kelly_fraction, stats = self.calculate_from_returns(returns)
            
            if stats.get("error"):
                return 0.0, 0.0
            
            # Simple confidence interval estimation
            # In production, would use more sophisticated bootstrap methods
            n = len(returns)
            std_error = math.sqrt(kelly_fraction * (1 - kelly_fraction) / n)
            
            # Z-score for confidence level
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_score = z_scores.get(confidence, 1.96)
            
            confidence_interval = z_score * std_error
            
            # Adjust Kelly fraction downward based on uncertainty
            adjusted_kelly = max(0.0, kelly_fraction - confidence_interval)
            
            return adjusted_kelly, confidence_interval
            
        except Exception as e:
            logger.error(f"Error calculating Kelly with confidence: {e}")
            return 0.0, 0.0


class KellyCalculator:
    """
    Advanced Kelly Criterion calculator with portfolio optimization.
    
    Provides portfolio-level Kelly calculation, multi-asset optimization,
    and dynamic position sizing based on changing market conditions.
    """
    
    def __init__(self):
        """Initialize Kelly calculator."""
        # AI-AGENT-REF: Advanced Kelly portfolio optimization
        self.kelly_criterion = KellyCriterion()
        # Use centralized config values instead of undefined KELLY_PARAMETERS
        self.lookback_periods = _DEFAULT_CONFIG.lookback_periods
        self.rebalance_frequency = _DEFAULT_CONFIG.rebalance_frequency
        
        # Store historical calculations
        self.calculation_history = []
        
        logger.info("KellyCalculator initialized")
    
    def calculate_portfolio_kelly(self, asset_returns: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate Kelly fractions for multiple assets in a portfolio.
        
        Args:
            asset_returns: Dictionary mapping asset symbols to return lists
            
        Returns:
            Dictionary mapping asset symbols to Kelly fractions
        """
        try:
            portfolio_kelly = {}
            total_kelly = 0.0
            
            for symbol, returns in asset_returns.items():
                kelly_fraction, stats = self.kelly_criterion.calculate_from_returns(returns)
                portfolio_kelly[symbol] = kelly_fraction
                total_kelly += kelly_fraction
                
                logger.debug(f"Kelly for {symbol}: {kelly_fraction:.3f} "
                           f"(trades: {stats.get('total_trades', 0)})")
            
            # Normalize if total Kelly exceeds maximum allowed
            if total_kelly > self.kelly_criterion.max_fraction:
                normalization_factor = self.kelly_criterion.max_fraction / total_kelly
                portfolio_kelly = {
                    symbol: fraction * normalization_factor 
                    for symbol, fraction in portfolio_kelly.items()
                }
                logger.info(f"Normalized portfolio Kelly fractions by factor {normalization_factor:.3f}")
            
            return portfolio_kelly
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Kelly: {e}")
            return {}
    
    def dynamic_kelly_adjustment(self, base_kelly: float, market_conditions: Dict) -> float:
        """
        Adjust Kelly fraction based on current market conditions.
        
        Args:
            base_kelly: Base Kelly fraction
            market_conditions: Dictionary of market condition indicators
            
        Returns:
            Adjusted Kelly fraction
        """
        try:
            adjusted_kelly = base_kelly
            
            # Volatility adjustment
            volatility = market_conditions.get('volatility', 0.2)
            if volatility > 0.3:  # High volatility
                adjusted_kelly *= 0.5  # Reduce position size
                logger.debug(f"High volatility adjustment: {volatility:.3f}")
            elif volatility < 0.1:  # Low volatility
                adjusted_kelly *= 1.2  # Slightly increase position size
                logger.debug(f"Low volatility adjustment: {volatility:.3f}")
            
            # Drawdown adjustment
            current_drawdown = market_conditions.get('drawdown', 0.0)
            if current_drawdown > 0.1:  # Significant drawdown
                drawdown_factor = 1 - (current_drawdown * 2)  # Reduce by 2x drawdown
                adjusted_kelly *= max(0.1, drawdown_factor)
                logger.debug(f"Drawdown adjustment: {current_drawdown:.3f}")
            
            # Market regime adjustment
            market_regime = market_conditions.get('regime', 'normal')
            if market_regime == 'crisis':
                adjusted_kelly *= 0.2  # Drastically reduce risk
                logger.debug("Crisis regime adjustment")
            elif market_regime == 'trending':
                adjusted_kelly *= 1.1  # Slightly increase risk
                logger.debug("Trending regime adjustment")
            
            # Ensure bounds
            adjusted_kelly = max(0.0, min(adjusted_kelly, self.kelly_criterion.max_fraction))
            
            return adjusted_kelly
            
        except Exception as e:
            logger.error(f"Error adjusting Kelly fraction: {e}")
            return base_kelly * 0.5  # Conservative fallback
    
    def kelly_with_correlation(self, asset_returns: Dict[str, List[float]], 
                              correlation_matrix: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Kelly fractions considering asset correlations.
        
        Args:
            asset_returns: Dictionary mapping asset symbols to return lists
            correlation_matrix: Optional correlation matrix between assets
            
        Returns:
            Dictionary mapping asset symbols to correlation-adjusted Kelly fractions
        """
        try:
            # Start with individual Kelly fractions
            individual_kelly = self.calculate_portfolio_kelly(asset_returns)
            
            if not correlation_matrix:
                logger.debug("No correlation matrix provided, using individual Kelly fractions")
                return individual_kelly
            
            # Adjust for correlations
            adjusted_kelly = {}
            for symbol in individual_kelly:
                kelly_fraction = individual_kelly[symbol]
                
                # Calculate correlation penalty
                correlation_penalty = 0.0
                for other_symbol in individual_kelly:
                    if other_symbol != symbol:
                        correlation = correlation_matrix.get(f"{symbol}_{other_symbol}", 0.0)
                        other_kelly = individual_kelly[other_symbol]
                        correlation_penalty += abs(correlation) * other_kelly
                
                # Apply penalty - reduce Kelly fraction based on correlation exposure
                penalty_factor = 1 - min(0.5, correlation_penalty)
                adjusted_kelly[symbol] = kelly_fraction * penalty_factor
                
                logger.debug(f"Correlation adjustment for {symbol}: "
                           f"{kelly_fraction:.3f} -> {adjusted_kelly[symbol]:.3f}")
            
            return adjusted_kelly
            
        except Exception as e:
            logger.error(f"Error calculating Kelly with correlation: {e}")
            return self.calculate_portfolio_kelly(asset_returns)
    
    def record_calculation(self, symbol: str, kelly_fraction: float, metadata: Dict):
        """Record Kelly calculation for historical analysis."""
        try:
            record = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "kelly_fraction": kelly_fraction,
                "metadata": metadata
            }
            
            self.calculation_history.append(record)
            
            # Keep only recent history
            cutoff_date = datetime.now() - timedelta(days=self.lookback_periods)
            self.calculation_history = [
                r for r in self.calculation_history 
                if r["timestamp"] >= cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error recording Kelly calculation: {e}")
    
    def get_calculation_history(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get historical Kelly calculations."""
        if symbol:
            return [r for r in self.calculation_history if r["symbol"] == symbol]
        return self.calculation_history.copy()