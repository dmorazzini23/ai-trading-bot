"""Kelly Criterion and advanced position sizing algorithms."""

from __future__ import annotations

import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

from ..core.models import (
    TradePosition, PortfolioMetrics, TradingSignal, 
    MarketData, StrategyPerformance
)
from ..core.enums import RiskLevel, TradingSide
from ..core.exceptions import RiskLimitExceededError


logger = logging.getLogger(__name__)


class KellyCriterion:
    """Kelly Criterion position sizing implementation."""
    
    def __init__(
        self,
        lookback_periods: int = 252,  # 1 year of daily data
        min_trades: int = 30,         # Minimum trades for calculation
        max_kelly: float = 0.25,      # Maximum Kelly fraction (25%)
        confidence_threshold: float = 0.6,  # Minimum confidence for sizing
    ):
        self.lookback_periods = lookback_periods
        self.min_trades = min_trades
        self.max_kelly = max_kelly
        self.confidence_threshold = confidence_threshold
        
        # Track strategy performance for Kelly calculation
        self._strategy_returns: Dict[str, List[float]] = {}
        self._trade_history: Dict[str, List[Dict]] = {}
    
    def calculate_kelly_fraction(
        self,
        strategy_id: str,
        signal: TradingSignal,
        performance: StrategyPerformance
    ) -> float:
        """Calculate optimal position size using Kelly Criterion.
        
        Args:
            strategy_id: Strategy identifier
            signal: Trading signal
            performance: Strategy performance metrics
            
        Returns:
            Kelly fraction (0.0 to max_kelly)
        """
        try:
            # Get strategy return history
            returns = self._strategy_returns.get(strategy_id, [])
            
            if len(returns) < self.min_trades:
                logger.warning(
                    f"Insufficient trade history for Kelly calculation: "
                    f"{len(returns)} < {self.min_trades}"
                )
                return self._conservative_sizing(signal)
            
            # Calculate win rate and average win/loss
            positive_returns = [r for r in returns if r > 0]
            negative_returns = [r for r in returns if r < 0]
            
            if not positive_returns or not negative_returns:
                logger.warning("No positive or negative returns for Kelly calculation")
                return self._conservative_sizing(signal)
            
            win_rate = len(positive_returns) / len(returns)
            avg_win = np.mean(positive_returns)
            avg_loss = abs(np.mean(negative_returns))
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss == 0:
                logger.warning("Average loss is zero, using conservative sizing")
                return self._conservative_sizing(signal)
            
            b = avg_win / avg_loss  # Odds received on a winning bet
            p = win_rate            # Probability of winning
            q = 1 - p              # Probability of losing
            
            kelly_fraction = (b * p - q) / b
            
            # Apply confidence and signal strength adjustments
            confidence_factor = min(signal.confidence, 1.0)
            strength_factor = min(signal.strength, 1.0)
            
            adjusted_kelly = kelly_fraction * confidence_factor * strength_factor
            
            # Cap at maximum Kelly fraction
            final_kelly = max(0.0, min(adjusted_kelly, self.max_kelly))
            
            logger.info(
                f"Kelly calculation for {strategy_id}: "
                f"raw={kelly_fraction:.4f}, "
                f"confidence={confidence_factor:.2f}, "
                f"strength={strength_factor:.2f}, "
                f"final={final_kelly:.4f}"
            )
            
            return final_kelly
            
        except Exception as e:
            logger.error(f"Kelly calculation failed: {e}")
            return self._conservative_sizing(signal)
    
    def _conservative_sizing(self, signal: TradingSignal) -> float:
        """Conservative position sizing when Kelly cannot be calculated.
        
        Args:
            signal: Trading signal
            
        Returns:
            Conservative position size fraction
        """
        base_size = 0.02  # 2% base position
        
        # Adjust based on signal quality
        if signal.confidence >= self.confidence_threshold:
            adjustment = signal.confidence * signal.strength
            return min(base_size * (1 + adjustment), 0.05)  # Max 5%
        
        return base_size * 0.5  # 1% for low confidence signals
    
    def update_performance(
        self,
        strategy_id: str,
        trade_return: float,
        trade_metadata: Optional[Dict] = None
    ) -> None:
        """Update strategy performance for Kelly calculation.
        
        Args:
            strategy_id: Strategy identifier
            trade_return: Trade return (percentage)
            trade_metadata: Additional trade information
        """
        if strategy_id not in self._strategy_returns:
            self._strategy_returns[strategy_id] = []
            self._trade_history[strategy_id] = []
        
        self._strategy_returns[strategy_id].append(trade_return)
        
        # Keep only recent history
        if len(self._strategy_returns[strategy_id]) > self.lookback_periods:
            self._strategy_returns[strategy_id] = (
                self._strategy_returns[strategy_id][-self.lookback_periods:]
            )
        
        # Store trade metadata
        trade_record = {
            'return': trade_return,
            'timestamp': datetime.now(timezone.utc),
            'metadata': trade_metadata or {}
        }
        self._trade_history[strategy_id].append(trade_record)
        
        if len(self._trade_history[strategy_id]) > self.lookback_periods:
            self._trade_history[strategy_id] = (
                self._trade_history[strategy_id][-self.lookback_periods:]
            )
    
    def get_strategy_stats(self, strategy_id: str) -> Dict[str, float]:
        """Get strategy statistics for Kelly calculation.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary of strategy statistics
        """
        returns = self._strategy_returns.get(strategy_id, [])
        
        if not returns:
            return {}
        
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        
        stats = {
            'total_trades': len(returns),
            'win_rate': len(positive_returns) / len(returns) if returns else 0,
            'avg_return': np.mean(returns),
            'avg_win': np.mean(positive_returns) if positive_returns else 0,
            'avg_loss': np.mean(negative_returns) if negative_returns else 0,
            'max_win': max(positive_returns) if positive_returns else 0,
            'max_loss': min(negative_returns) if negative_returns else 0,
            'volatility': np.std(returns) if len(returns) > 1 else 0,
            'sharpe_ratio': (
                np.mean(returns) / np.std(returns) 
                if len(returns) > 1 and np.std(returns) > 0 else 0
            )
        }
        
        return stats


class VolatilityPositionSizing:
    """Volatility-based position sizing."""
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% target volatility
        lookback_days: int = 20,          # 20-day volatility calculation
        min_position: float = 0.005,      # 0.5% minimum position
        max_position: float = 0.10,       # 10% maximum position
    ):
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        self.min_position = min_position
        self.max_position = max_position
    
    def calculate_position_size(
        self,
        symbol: str,
        historical_data: List[MarketData],
        portfolio_value: Decimal,
        signal_strength: float = 1.0
    ) -> float:
        """Calculate position size based on volatility targeting.
        
        Args:
            symbol: Trading symbol
            historical_data: Recent price data
            portfolio_value: Current portfolio value
            signal_strength: Signal strength multiplier
            
        Returns:
            Position size as fraction of portfolio
        """
        try:
            if len(historical_data) < self.lookback_days:
                logger.warning(
                    f"Insufficient data for volatility calculation: "
                    f"{len(historical_data)} < {self.lookback_days}"
                )
                return self.min_position
            
            # Calculate returns
            prices = [float(data.close_price) for data in historical_data[-self.lookback_days:]]
            returns = np.diff(np.log(prices))
            
            if len(returns) == 0:
                return self.min_position
            
            # Calculate annualized volatility
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)  # Annualize
            
            if annual_vol == 0:
                return self.min_position
            
            # Position size = target_vol / asset_vol
            base_position = self.target_volatility / annual_vol
            
            # Apply signal strength
            adjusted_position = base_position * signal_strength
            
            # Apply bounds
            final_position = max(
                self.min_position,
                min(adjusted_position, self.max_position)
            )
            
            logger.debug(
                f"Volatility sizing for {symbol}: "
                f"vol={annual_vol:.3f}, "
                f"base={base_position:.3f}, "
                f"final={final_position:.3f}"
            )
            
            return final_position
            
        except Exception as e:
            logger.error(f"Volatility position sizing failed for {symbol}: {e}")
            return self.min_position


class RiskParityPositionSizing:
    """Risk parity position sizing across portfolio."""
    
    def __init__(
        self,
        lookback_days: int = 60,
        rebalance_frequency: int = 5,  # Days between rebalancing
    ):
        self.lookback_days = lookback_days
        self.rebalance_frequency = rebalance_frequency
        self._last_rebalance = None
        self._correlation_matrix = None
        self._volatilities = {}
    
    def calculate_risk_parity_weights(
        self,
        symbols: List[str],
        historical_data: Dict[str, List[MarketData]]
    ) -> Dict[str, float]:
        """Calculate risk parity weights for portfolio.
        
        Args:
            symbols: List of trading symbols
            historical_data: Historical price data by symbol
            
        Returns:
            Risk parity weights by symbol
        """
        try:
            # Check if rebalancing is needed
            if (self._last_rebalance is None or 
                (datetime.now(timezone.utc) - self._last_rebalance).days >= self.rebalance_frequency):
                
                self._update_risk_metrics(symbols, historical_data)
                self._last_rebalance = datetime.now(timezone.utc)
            
            # Equal risk contribution = 1/volatility (simplified)
            weights = {}
            total_inv_vol = 0
            
            for symbol in symbols:
                vol = self._volatilities.get(symbol, 0.15)  # Default 15% vol
                inv_vol = 1.0 / vol if vol > 0 else 0
                weights[symbol] = inv_vol
                total_inv_vol += inv_vol
            
            # Normalize weights
            if total_inv_vol > 0:
                for symbol in weights:
                    weights[symbol] /= total_inv_vol
            else:
                # Equal weights fallback
                equal_weight = 1.0 / len(symbols) if symbols else 0
                weights = {symbol: equal_weight for symbol in symbols}
            
            return weights
            
        except Exception as e:
            logger.error(f"Risk parity calculation failed: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / len(symbols) if symbols else 0
            return {symbol: equal_weight for symbol in symbols}
    
    def _update_risk_metrics(
        self,
        symbols: List[str],
        historical_data: Dict[str, List[MarketData]]
    ) -> None:
        """Update volatilities and correlations."""
        returns_data = {}
        
        # Calculate returns for each symbol
        for symbol in symbols:
            data = historical_data.get(symbol, [])
            if len(data) >= self.lookback_days:
                prices = [float(d.close_price) for d in data[-self.lookback_days:]]
                returns = np.diff(np.log(prices))
                returns_data[symbol] = returns
                
                # Store volatility
                daily_vol = np.std(returns) if len(returns) > 1 else 0.15
                self._volatilities[symbol] = daily_vol * np.sqrt(252)  # Annualize
        
        # Calculate correlation matrix
        if len(returns_data) > 1:
            symbols_with_data = list(returns_data.keys())
            min_length = min(len(returns_data[s]) for s in symbols_with_data)
            
            returns_matrix = np.array([
                returns_data[s][:min_length] for s in symbols_with_data
            ])
            
            self._correlation_matrix = np.corrcoef(returns_matrix)
        
        logger.info(f"Updated risk metrics for {len(symbols)} symbols")


class PositionSizer:
    """Unified position sizing system combining multiple approaches."""
    
    def __init__(
        self,
        kelly_criterion: Optional[KellyCriterion] = None,
        volatility_sizer: Optional[VolatilityPositionSizing] = None,
        risk_parity: Optional[RiskParityPositionSizing] = None,
        default_method: str = "kelly",
        max_portfolio_exposure: float = 0.95,
    ):
        self.kelly_criterion = kelly_criterion or KellyCriterion()
        self.volatility_sizer = volatility_sizer or VolatilityPositionSizing()
        self.risk_parity = risk_parity or RiskParityPositionSizing()
        self.default_method = default_method
        self.max_portfolio_exposure = max_portfolio_exposure
    
    def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_metrics: PortfolioMetrics,
        current_positions: Dict[str, TradePosition],
        historical_data: Optional[Dict[str, List[MarketData]]] = None,
        strategy_performance: Optional[StrategyPerformance] = None,
        method: Optional[str] = None
    ) -> Decimal:
        """Calculate optimal position size using specified method.
        
        Args:
            signal: Trading signal
            portfolio_metrics: Current portfolio state
            current_positions: Existing positions
            historical_data: Historical market data
            strategy_performance: Strategy performance metrics
            method: Sizing method ('kelly', 'volatility', 'risk_parity', or 'combined')
            
        Returns:
            Position size in portfolio units
        """
        method = method or self.default_method
        
        try:
            # Check current exposure
            current_exposure = self._calculate_current_exposure(
                current_positions, portfolio_metrics.total_value
            )
            
            if current_exposure >= self.max_portfolio_exposure:
                logger.warning(
                    f"Portfolio exposure limit reached: {current_exposure:.2%} >= "
                    f"{self.max_portfolio_exposure:.2%}"
                )
                return Decimal('0')
            
            # Calculate base position size
            if method == "kelly" and strategy_performance:
                base_size = self.kelly_criterion.calculate_kelly_fraction(
                    strategy_performance.strategy_id,
                    signal,
                    strategy_performance
                )
            elif method == "volatility" and historical_data:
                symbol_data = historical_data.get(signal.symbol, [])
                base_size = self.volatility_sizer.calculate_position_size(
                    signal.symbol,
                    symbol_data,
                    portfolio_metrics.total_value,
                    signal.strength
                )
            elif method == "combined" and historical_data and strategy_performance:
                # Combine Kelly and volatility sizing
                kelly_size = self.kelly_criterion.calculate_kelly_fraction(
                    strategy_performance.strategy_id,
                    signal,
                    strategy_performance
                )
                
                symbol_data = historical_data.get(signal.symbol, [])
                vol_size = self.volatility_sizer.calculate_position_size(
                    signal.symbol,
                    symbol_data,
                    portfolio_metrics.total_value,
                    signal.strength
                )
                
                # Weight average (60% Kelly, 40% volatility)
                base_size = 0.6 * kelly_size + 0.4 * vol_size
            else:
                # Conservative default
                base_size = 0.02 * signal.strength * signal.confidence
            
            # Apply portfolio constraints
            max_additional = self.max_portfolio_exposure - current_exposure
            final_size = min(base_size, max_additional)
            
            # Convert to dollar amount and back to fraction
            dollar_size = float(final_size) * float(portfolio_metrics.total_value)
            
            logger.info(
                f"Position sizing for {signal.symbol}: "
                f"method={method}, base={base_size:.3f}, "
                f"final={final_size:.3f}, amount=${dollar_size:,.0f}"
            )
            
            return Decimal(str(final_size))
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            # Return conservative size
            return Decimal('0.01')  # 1% fallback
    
    def _calculate_current_exposure(
        self,
        positions: Dict[str, TradePosition],
        portfolio_value: Decimal
    ) -> float:
        """Calculate current portfolio exposure.
        
        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            
        Returns:
            Current exposure as fraction of portfolio
        """
        if not positions or portfolio_value <= 0:
            return 0.0
        
        total_exposure = sum(
            abs(float(pos.market_value)) for pos in positions.values()
        )
        
        return total_exposure / float(portfolio_value)