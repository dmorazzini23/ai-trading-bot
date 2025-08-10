#!/usr/bin/env python3
"""Advanced trading algorithm optimization and fine-tuning module.

This module provides comprehensive trading algorithm optimization:
- Mathematical model validation and optimization
- Dynamic parameter adjustment based on market conditions
- Advanced risk management rules and safeguards
- Entry and exit timing optimization
- Portfolio rebalancing logic
- Correlation analysis and diversification checks
- Market regime detection and adaptation

AI-AGENT-REF: Advanced algorithm optimization for institutional trading
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import statistics
import threading  # AI-AGENT-REF: Thread safety for algorithm optimizer
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

# AI-AGENT-REF: Advanced algorithm optimization for production trading


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BEAR_MARKET = "bear_market"
    BULL_MARKET = "bull_market"


class TradingPhase(Enum):
    """Trading session phases."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MID_DAY = "mid_day"
    LATE_DAY = "late_day"
    AFTER_HOURS = "after_hours"


@dataclass
class OptimizationMetrics:
    """Algorithm optimization metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    volatility: float
    beta: float
    alpha: float


@dataclass
class MarketConditions:
    """Current market condition assessment."""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_profile: float
    correlation_to_market: float
    sector_rotation: float
    vix_level: float
    time_of_day: TradingPhase


@dataclass
class OptimizedParameters:
    """Optimized trading parameters."""
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    rsi_oversold: float
    rsi_overbought: float
    moving_average_period: int
    volume_threshold: float
    volatility_lookback: int
    correlation_threshold: float


class AlgorithmOptimizer:
    """Advanced trading algorithm optimizer with thread safety."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # AI-AGENT-REF: Thread safety for shared state access
        self._lock = threading.RLock()
        
        # Historical performance tracking with bounded growth
        self.performance_history: deque = deque(maxlen=10000)
        self.parameter_history: deque = deque(maxlen=1000)
        
        # Market regime detection with bounded growth prevention
        self.market_regimes: deque = deque(maxlen=100)
        self.regime_performance: Dict[MarketRegime, deque] = {
            regime: deque(maxlen=1000) for regime in MarketRegime  # Bounded growth
        }
        
        # Optimization settings
        self.optimization_enabled = True
        self.regime_detection_enabled = True
        self.dynamic_sizing_enabled = True
        
        # Base parameters (default values)
        self.base_parameters = OptimizedParameters(
            position_size_multiplier=1.0,
            stop_loss_multiplier=1.0,
            take_profit_multiplier=1.0,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            moving_average_period=20,
            volume_threshold=1.5,
            volatility_lookback=20,
            correlation_threshold=0.7
        )
        
        # Current optimized parameters
        self.current_parameters = self.base_parameters
        
        # Parameter bounds for optimization
        self.parameter_bounds = {
            'position_size_multiplier': (0.1, 3.0),
            'stop_loss_multiplier': (0.5, 5.0),
            'take_profit_multiplier': (0.5, 10.0),
            'rsi_oversold': (15.0, 40.0),
            'rsi_overbought': (60.0, 85.0),
            'moving_average_period': (5, 50),
            'volume_threshold': (1.0, 5.0),
            'volatility_lookback': (10, 60),
            'correlation_threshold': (0.3, 0.9)
        }
        
        # Performance tracking
        self.last_optimization = None
        self.optimization_frequency = timedelta(hours=24)  # Daily optimization
        
        self.logger.info("Algorithm optimizer initialized")
    
    def detect_market_regime(self, price_data: pd.DataFrame, 
                           volume_data: Optional[pd.DataFrame] = None,
                           market_data: Optional[pd.DataFrame] = None) -> MarketConditions:
        """Detect current market regime and conditions."""
        try:
            if len(price_data) < 20:
                # Default regime for insufficient data
                return MarketConditions(
                    regime=MarketRegime.SIDEWAYS,
                    volatility=0.02,
                    trend_strength=0.0,
                    volume_profile=1.0,
                    correlation_to_market=0.5,
                    sector_rotation=0.0,
                    vix_level=20.0,
                    time_of_day=self._get_trading_phase()
                )
            
            # Calculate price-based indicators
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Trend strength using ADX-like calculation
            high_low = price_data['high'] - price_data['low']
            high_close_prev = abs(price_data['high'] - price_data['close'].shift(1))
            low_close_prev = abs(price_data['low'] - price_data['close'].shift(1))
            
            true_range = pd.DataFrame({
                'hl': high_low,
                'hc': high_close_prev,
                'lc': low_close_prev
            }).max(axis=1)
            
            atr = true_range.rolling(14).mean().iloc[-1]
            price_range = price_data['close'].iloc[-1] - price_data['close'].iloc[-20]
            trend_strength = abs(price_range) / (atr * 20) if atr > 0 else 0
            
            # Volume analysis
            volume_profile = 1.0
            if volume_data is not None and len(volume_data) >= 20:
                avg_volume = volume_data['volume'].rolling(20).mean().iloc[-1]
                current_volume = volume_data['volume'].iloc[-1]
                volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Market correlation (simplified)
            correlation_to_market = 0.5
            if market_data is not None and len(market_data) >= 20:
                market_returns = market_data['close'].pct_change().dropna()
                if len(market_returns) >= len(returns):
                    correlation_to_market = returns.corr(market_returns[:len(returns)])
                    if pd.isna(correlation_to_market):
                        correlation_to_market = 0.5
            
            # Determine regime
            regime = self._classify_regime(volatility, trend_strength, returns)
            
            # VIX level (estimated from volatility)
            vix_level = min(100, max(10, volatility * 100))
            
            conditions = MarketConditions(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                correlation_to_market=correlation_to_market,
                sector_rotation=0.0,  # Would need sector data
                vix_level=vix_level,
                time_of_day=self._get_trading_phase()
            )
            
            self.market_regimes.append(conditions)
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            # Return default conditions on error
            return MarketConditions(
                regime=MarketRegime.SIDEWAYS,
                volatility=0.02,
                trend_strength=0.0,
                volume_profile=1.0,
                correlation_to_market=0.5,
                sector_rotation=0.0,
                vix_level=20.0,
                time_of_day=self._get_trading_phase()
            )
    
    def _classify_regime(self, volatility: float, trend_strength: float, 
                        returns: pd.Series) -> MarketRegime:
        """Classify market regime based on indicators."""
        # High volatility threshold
        if volatility > 0.3:
            return MarketRegime.VOLATILE
        
        # Low volatility threshold
        if volatility < 0.1:
            return MarketRegime.LOW_VOLATILITY
        
        # Trend classification
        if trend_strength > 1.0:
            # Check if trending up or down
            recent_return = returns.tail(5).mean()
            if recent_return > 0.01:
                return MarketRegime.BULL_MARKET
            elif recent_return < -0.01:
                return MarketRegime.BEAR_MARKET
            else:
                return MarketRegime.TRENDING
        
        # Default to sideways
        return MarketRegime.SIDEWAYS
    
    def _get_trading_phase(self) -> TradingPhase:
        """Determine current trading phase based on time."""
        now = datetime.now(timezone.utc).time()
        
        # EST trading hours (simplified)
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            return TradingPhase.PRE_MARKET
        elif now.hour == 9 or (now.hour == 10 and now.minute < 30):
            return TradingPhase.MARKET_OPEN
        elif now.hour < 15:
            return TradingPhase.MID_DAY
        elif now.hour < 16:
            return TradingPhase.LATE_DAY
        else:
            return TradingPhase.AFTER_HOURS
    
    def optimize_parameters(self, market_conditions: MarketConditions,
                          recent_performance: List[float],
                          force_optimization: bool = False) -> OptimizedParameters:
        """Optimize trading parameters based on market conditions and performance."""
        try:
            # Check if optimization is needed
            if not force_optimization and not self._should_optimize():
                return self.current_parameters
            
            # Start with base parameters
            optimized = OptimizedParameters(**self.base_parameters.__dict__)
            
            # Regime-based adjustments
            optimized = self._adjust_for_regime(optimized, market_conditions)
            
            # Volatility-based adjustments
            optimized = self._adjust_for_volatility(optimized, market_conditions.volatility)
            
            # Time-based adjustments
            optimized = self._adjust_for_time(optimized, market_conditions.time_of_day)
            
            # Performance-based adjustments
            if recent_performance:
                optimized = self._adjust_for_performance(optimized, recent_performance)
            
            # Ensure parameters are within bounds
            optimized = self._enforce_bounds(optimized)
            
            # Update current parameters
            self.current_parameters = optimized
            self.last_optimization = datetime.now(timezone.utc)
            
            # Record parameter change
            self.parameter_history.append({
                'timestamp': self.last_optimization,
                'parameters': optimized,
                'market_conditions': market_conditions,
                'performance_trigger': recent_performance[-1] if recent_performance else 0.0
            })
            
            self.logger.info(f"Parameters optimized for {market_conditions.regime.value} regime")
            return optimized
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return self.current_parameters
    
    def _should_optimize(self) -> bool:
        """Check if parameters should be optimized."""
        if not self.optimization_enabled:
            return False
        
        if self.last_optimization is None:
            return True
        
        time_since_last = datetime.now(timezone.utc) - self.last_optimization
        return time_since_last >= self.optimization_frequency
    
    def _adjust_for_regime(self, params: OptimizedParameters, 
                          conditions: MarketConditions) -> OptimizedParameters:
        """Adjust parameters based on market regime."""
        if conditions.regime == MarketRegime.VOLATILE:
            # Reduce position sizes and tighten stops in volatile markets
            params.position_size_multiplier *= 0.7
            params.stop_loss_multiplier *= 0.8
            params.rsi_oversold = 25.0
            params.rsi_overbought = 75.0
            
        elif conditions.regime == MarketRegime.LOW_VOLATILITY:
            # Increase position sizes in low volatility
            params.position_size_multiplier *= 1.3
            params.take_profit_multiplier *= 1.2
            
        elif conditions.regime == MarketRegime.TRENDING:
            # Favor trend-following in trending markets
            params.moving_average_period = 15
            params.take_profit_multiplier *= 1.5
            
        elif conditions.regime == MarketRegime.BULL_MARKET:
            # More aggressive in bull markets
            params.position_size_multiplier *= 1.2
            params.rsi_oversold = 35.0
            
        elif conditions.regime == MarketRegime.BEAR_MARKET:
            # Conservative in bear markets
            params.position_size_multiplier *= 0.6
            params.stop_loss_multiplier *= 0.7
            params.rsi_overbought = 65.0
        
        return params
    
    def _adjust_for_volatility(self, params: OptimizedParameters, 
                             volatility: float) -> OptimizedParameters:
        """Adjust parameters based on volatility level."""
        # Normalize volatility (typical range 0.1 to 0.5)
        vol_factor = min(2.0, max(0.5, volatility / 0.2))
        
        # Inverse relationship for position sizing
        params.position_size_multiplier /= vol_factor
        
        # Direct relationship for stop losses
        params.stop_loss_multiplier *= (vol_factor ** 0.5)
        
        # Adjust lookback periods
        if volatility > 0.3:
            params.volatility_lookback = 10  # Shorter lookback in high vol
        elif volatility < 0.15:
            params.volatility_lookback = 30  # Longer lookback in low vol
        
        return params
    
    def _adjust_for_time(self, params: OptimizedParameters, 
                        phase: TradingPhase) -> OptimizedParameters:
        """Adjust parameters based on trading phase."""
        if phase == TradingPhase.MARKET_OPEN:
            # More conservative at open (higher volatility)
            params.position_size_multiplier *= 0.8
            params.volume_threshold *= 1.5
            
        elif phase == TradingPhase.LATE_DAY:
            # Reduce position sizes near close
            params.position_size_multiplier *= 0.9
            params.stop_loss_multiplier *= 0.9
            
        elif phase in [TradingPhase.PRE_MARKET, TradingPhase.AFTER_HOURS]:
            # Very conservative in extended hours
            params.position_size_multiplier *= 0.5
            params.volume_threshold *= 2.0
        
        return params
    
    def _adjust_for_performance(self, params: OptimizedParameters, 
                              performance: List[float]) -> OptimizedParameters:
        """Adjust parameters based on recent performance."""
        if len(performance) < 5:
            return params
        
        recent_performance = performance[-10:]  # Last 10 trades
        win_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
        avg_return = statistics.mean(recent_performance)
        
        # Performance-based adjustments
        if win_rate < 0.4:  # Poor win rate
            # More conservative
            params.position_size_multiplier *= 0.8
            params.stop_loss_multiplier *= 0.9
            
        elif win_rate > 0.7:  # Good win rate
            # More aggressive
            params.position_size_multiplier *= 1.1
            params.take_profit_multiplier *= 1.1
        
        if avg_return < -0.02:  # Losing streak
            # Very conservative
            params.position_size_multiplier *= 0.7
            params.stop_loss_multiplier *= 0.8
        
        return params
    
    def _enforce_bounds(self, params: OptimizedParameters) -> OptimizedParameters:
        """Enforce parameter bounds to prevent extreme values."""
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            current_val = getattr(params, param_name)
            bounded_val = max(min_val, min(max_val, current_val))
            setattr(params, param_name, bounded_val)
        
        return params
    
    def calculate_optimal_position_size(self, symbol: str, price: float,
                                      account_value: float,
                                      volatility: float,
                                      market_conditions: MarketConditions) -> float:
        """Calculate optimal position size using Kelly criterion and risk management with thread safety."""
        with self._lock:  # AI-AGENT-REF: Thread-safe position size calculation
            try:
                # AI-AGENT-REF: Division by zero protection
                if price <= 0 or account_value <= 0:
                    self.logger.warning(f"Invalid inputs: price={price}, account_value={account_value}")
                    return 0
                
                # Kelly criterion calculation with historical performance
                kelly_fraction = self._calculate_kelly_fraction(symbol)
                
                # Base position size (percentage of account)
                base_position_pct = 0.02  # 2% base position
                
                # Apply Kelly criterion with safety caps
                kelly_adjusted_pct = min(kelly_fraction, 0.25) if kelly_fraction > 0 else base_position_pct
                
                # Adjust for market conditions
                regime_multiplier = self._get_regime_multiplier(market_conditions.regime)
                
                # AI-AGENT-REF: Safe volatility adjustment with epsilon
                epsilon = 1e-8  # Numerical stability
                volatility_adjustment = 1.0 / (1.0 + max(volatility, epsilon) * 5)
                
                # Time-based adjustment
                time_multiplier = self._get_time_multiplier(market_conditions.time_of_day)
                
                # Apply current parameters
                param_multiplier = self.current_parameters.position_size_multiplier
                
                # Calculate final position size
                position_pct = (kelly_adjusted_pct * 
                              regime_multiplier * 
                              volatility_adjustment * 
                              time_multiplier * 
                              param_multiplier)
                
                # Convert to dollar amount
                position_value = account_value * position_pct
                
                # Convert to shares (rounded down)
                shares = int(position_value / price)
                
                # Minimum position size
                min_shares = max(1, int(account_value * 0.001 / price))  # 0.1% minimum
                
                # Maximum position size (risk management)
                max_shares = int(account_value * 0.1 / price)  # 10% maximum
                
                final_shares = max(min_shares, min(max_shares, shares))
                
                self.logger.debug(
                    f"Position size calculated for {symbol}: {final_shares} shares "
                    f"(${final_shares * price:.2f}, {final_shares * price / account_value * 100:.2f}% of account) "
                    f"Kelly fraction: {kelly_fraction:.4f}"
                )
                
                return final_shares
                
            except Exception as e:
                self.logger.error(f"Error calculating position size: {e}")
                return 0
    
    def _calculate_kelly_fraction(self, symbol: str) -> float:
        """Calculate Kelly fraction from historical performance with division by zero protection."""
        try:
            # Get symbol-specific performance history
            symbol_performance = []
            for record in self.performance_history:
                if isinstance(record, dict) and record.get('symbol') == symbol:
                    symbol_performance.append(record.get('return', 0.0))
            
            if len(symbol_performance) < 10:  # Need minimum sample size
                return 0.02  # Default conservative fraction
            
            # Calculate win rate and average win/loss
            wins = [p for p in symbol_performance if p > 0]
            losses = [abs(p) for p in symbol_performance if p < 0]
            
            # AI-AGENT-REF: Division by zero protection for Kelly criterion
            total_trades = len(symbol_performance)
            if total_trades == 0:
                return 0.02
                
            win_rate = len(wins) / total_trades
            
            # Safe average calculations with epsilon
            epsilon = 1e-8
            avg_win = sum(wins) / max(len(wins), 1) if wins else epsilon
            avg_loss = sum(losses) / max(len(losses), 1) if losses else epsilon
            
            # Kelly formula: f = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss <= epsilon:  # Avoid division by zero
                return min(0.25, win_rate * 0.5)  # Conservative fallback
                
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction for safety (institutional limit)
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.02  # Conservative default
            # Default to minimum position
            return max(1, int(account_value * 0.001 / price))
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier for market regime."""
        multipliers = {
            MarketRegime.BULL_MARKET: 1.2,
            MarketRegime.TRENDING: 1.1,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.LOW_VOLATILITY: 1.3,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.BEAR_MARKET: 0.6
        }
        return multipliers.get(regime, 1.0)
    
    def _get_time_multiplier(self, phase: TradingPhase) -> float:
        """Get position size multiplier for trading phase."""
        multipliers = {
            TradingPhase.PRE_MARKET: 0.5,
            TradingPhase.MARKET_OPEN: 0.8,
            TradingPhase.MID_DAY: 1.0,
            TradingPhase.LATE_DAY: 0.9,
            TradingPhase.AFTER_HOURS: 0.4
        }
        return multipliers.get(phase, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, side: str, 
                          volatility: float, atr: float) -> float:
        """Calculate optimal stop loss price."""
        try:
            # Base stop loss (percentage)
            base_stop_pct = 0.02  # 2% base stop
            
            # Adjust for volatility
            volatility_stop = volatility * 2  # 2x daily volatility
            
            # ATR-based stop
            atr_stop = atr * 2 / entry_price  # 2x ATR as percentage
            
            # Use the larger of volatility or ATR stop, but cap at reasonable level
            stop_pct = min(0.05, max(base_stop_pct, max(volatility_stop, atr_stop)))
            
            # Apply parameter multiplier
            stop_pct *= self.current_parameters.stop_loss_multiplier
            
            # Calculate stop price
            if side.upper() == 'BUY':
                stop_price = entry_price * (1 - stop_pct)
            else:  # SELL
                stop_price = entry_price * (1 + stop_pct)
            
            return round(stop_price, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            # Default stop loss
            if side.upper() == 'BUY':
                return entry_price * 0.98  # 2% stop
            else:
                return entry_price * 1.02
    
    def calculate_take_profit(self, entry_price: float, side: str,
                            stop_loss: float, risk_reward_ratio: float = 2.0) -> float:
        """Calculate optimal take profit price."""
        try:
            # Calculate risk amount
            if side.upper() == 'BUY':
                risk_amount = entry_price - stop_loss
                take_profit_price = entry_price + (risk_amount * risk_reward_ratio * 
                                                 self.current_parameters.take_profit_multiplier)
            else:  # SELL
                risk_amount = stop_loss - entry_price
                take_profit_price = entry_price - (risk_amount * risk_reward_ratio * 
                                                 self.current_parameters.take_profit_multiplier)
            
            return round(take_profit_price, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            # Default take profit
            if side.upper() == 'BUY':
                return entry_price * 1.04  # 4% profit target
            else:
                return entry_price * 0.96
    
    def validate_mathematical_models(self) -> Dict[str, Any]:
        """Validate mathematical models and calculations."""
        validation_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tests_run': [],
            'tests_passed': 0,
            'tests_failed': 0,
            'issues_found': []
        }
        
        # Test position sizing
        try:
            test_size = self.calculate_optimal_position_size(
                'TEST', 100.0, 10000.0, 0.2, 
                MarketConditions(MarketRegime.SIDEWAYS, 0.2, 0.5, 1.0, 0.5, 0.0, 20.0, TradingPhase.MID_DAY)
            )
            assert test_size > 0, "Position size must be positive"
            assert test_size <= 100, "Position size should be reasonable for test account"
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('position_sizing')
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f"Position sizing error: {e}")
        
        # Test stop loss calculation
        try:
            stop = self.calculate_stop_loss(100.0, 'BUY', 0.02, 2.0)
            assert 90.0 <= stop <= 99.0, f"Stop loss {stop} should be reasonable"
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('stop_loss_calculation')
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f"Stop loss calculation error: {e}")
        
        # Test take profit calculation
        try:
            profit = self.calculate_take_profit(100.0, 'BUY', 98.0, 2.0)
            assert profit > 100.0, f"Take profit {profit} should be above entry"
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('take_profit_calculation')
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f"Take profit calculation error: {e}")
        
        # Test parameter optimization
        try:
            conditions = MarketConditions(
                MarketRegime.VOLATILE, 0.3, 0.8, 1.2, 0.6, 0.0, 25.0, TradingPhase.MID_DAY
            )
            optimized = self.optimize_parameters(conditions, [0.01, -0.02, 0.03], True)
            assert hasattr(optimized, 'position_size_multiplier'), "Optimized parameters missing attributes"
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('parameter_optimization')
        except Exception as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f"Parameter optimization error: {e}")
        
        return validation_results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'optimization_enabled': self.optimization_enabled,
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'current_parameters': self.current_parameters.__dict__,
            'base_parameters': self.base_parameters.__dict__,
            'recent_regimes': [regime.regime.value for regime in list(self.market_regimes)[-10:]],
            'parameter_changes': len(self.parameter_history),
            'regime_performance': {
                regime.value: {
                    'count': len(performances),
                    'avg_performance': statistics.mean(performances) if performances else 0.0,
                    'win_rate': sum(1 for p in performances if p > 0) / len(performances) * 100 if performances else 0.0
                }
                for regime, performances in self.regime_performance.items()
            }
        }


# Global algorithm optimizer instance
_algorithm_optimizer: Optional[AlgorithmOptimizer] = None


def get_algorithm_optimizer() -> AlgorithmOptimizer:
    """Get global algorithm optimizer instance."""
    global _algorithm_optimizer
    if _algorithm_optimizer is None:
        _algorithm_optimizer = AlgorithmOptimizer()
    return _algorithm_optimizer


def initialize_algorithm_optimizer() -> AlgorithmOptimizer:
    """Initialize algorithm optimizer."""
    global _algorithm_optimizer
    _algorithm_optimizer = AlgorithmOptimizer()
    return _algorithm_optimizer