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
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = 'trending'
    SIDEWAYS = 'sideways'
    VOLATILE = 'volatile'
    LOW_VOLATILITY = 'low_volatility'
    BEAR_MARKET = 'bear_market'
    BULL_MARKET = 'bull_market'

class TradingPhase(Enum):
    """Trading session phases."""
    PRE_MARKET = 'pre_market'
    MARKET_OPEN = 'market_open'
    MID_DAY = 'mid_day'
    LATE_DAY = 'late_day'
    AFTER_HOURS = 'after_hours'

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
        self._lock = threading.RLock()
        self.performance_history: deque = deque(maxlen=10000)
        self.parameter_history: deque = deque(maxlen=1000)
        self.market_regimes: deque = deque(maxlen=100)
        self.regime_performance: dict[MarketRegime, deque] = {regime: deque(maxlen=1000) for regime in MarketRegime}
        self.optimization_enabled = True
        self.regime_detection_enabled = True
        self.dynamic_sizing_enabled = True
        self.base_parameters = OptimizedParameters(position_size_multiplier=1.0, stop_loss_multiplier=1.0, take_profit_multiplier=1.0, rsi_oversold=30.0, rsi_overbought=70.0, moving_average_period=20, volume_threshold=1.5, volatility_lookback=20, correlation_threshold=0.7)
        self.current_parameters = self.base_parameters
        self.parameter_bounds = {'position_size_multiplier': (0.1, 3.0), 'stop_loss_multiplier': (0.5, 5.0), 'take_profit_multiplier': (0.5, 10.0), 'rsi_oversold': (15.0, 40.0), 'rsi_overbought': (60.0, 85.0), 'moving_average_period': (5, 50), 'volume_threshold': (1.0, 5.0), 'volatility_lookback': (10, 60), 'correlation_threshold': (0.3, 0.9)}
        self.last_optimization = None
        self.optimization_frequency = timedelta(hours=24)
        self.logger.info('Algorithm optimizer initialized')

    def detect_market_regime(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None=None, market_data: pd.DataFrame | None=None) -> MarketConditions:
        """Detect current market regime and conditions."""
        try:
            if len(price_data) < 20:
                return MarketConditions(regime=MarketRegime.SIDEWAYS, volatility=0.02, trend_strength=0.0, volume_profile=1.0, correlation_to_market=0.5, sector_rotation=0.0, vix_level=20.0, time_of_day=self._get_trading_phase())
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            high_low = price_data['high'] - price_data['low']
            high_close_prev = abs(price_data['high'] - price_data['close'].shift(1))
            low_close_prev = abs(price_data['low'] - price_data['close'].shift(1))
            true_range = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            price_range = price_data['close'].iloc[-1] - price_data['close'].iloc[-20]
            trend_strength = abs(price_range) / (atr * 20) if atr > 0 else 0
            volume_profile = 1.0
            if volume_data is not None and len(volume_data) >= 20:
                avg_volume = volume_data['volume'].rolling(20).mean().iloc[-1]
                current_volume = volume_data['volume'].iloc[-1]
                volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0
            correlation_to_market = 0.5
            if market_data is not None and len(market_data) >= 20:
                market_returns = market_data['close'].pct_change().dropna()
                if len(market_returns) >= len(returns):
                    correlation_to_market = returns.corr(market_returns[:len(returns)])
                    if pd.isna(correlation_to_market):
                        correlation_to_market = 0.5
            regime = self._classify_regime(volatility, trend_strength, returns)
            vix_level = min(100, max(10, volatility * 100))
            conditions = MarketConditions(regime=regime, volatility=volatility, trend_strength=trend_strength, volume_profile=volume_profile, correlation_to_market=correlation_to_market, sector_rotation=0.0, vix_level=vix_level, time_of_day=self._get_trading_phase())
            self.market_regimes.append(conditions)
            return conditions
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error detecting market regime: {e}')
            return MarketConditions(regime=MarketRegime.SIDEWAYS, volatility=0.02, trend_strength=0.0, volume_profile=1.0, correlation_to_market=0.5, sector_rotation=0.0, vix_level=20.0, time_of_day=self._get_trading_phase())

    def _classify_regime(self, volatility: float, trend_strength: float, returns: pd.Series) -> MarketRegime:
        """Classify market regime based on indicators."""
        if volatility > 0.3:
            return MarketRegime.VOLATILE
        if volatility < 0.1:
            return MarketRegime.LOW_VOLATILITY
        if trend_strength > 1.0:
            recent_return = returns.tail(5).mean()
            if recent_return > 0.01:
                return MarketRegime.BULL_MARKET
            elif recent_return < -0.01:
                return MarketRegime.BEAR_MARKET
            else:
                return MarketRegime.TRENDING
        return MarketRegime.SIDEWAYS

    def _get_trading_phase(self) -> TradingPhase:
        """Determine current trading phase based on time."""
        now = datetime.now(UTC).time()
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

    def optimize_parameters(self, market_conditions: MarketConditions, recent_performance: list[float], force_optimization: bool=False) -> OptimizedParameters:
        """Optimize trading parameters based on market conditions and performance."""
        try:
            if not force_optimization and (not self._should_optimize()):
                return self.current_parameters
            optimized = OptimizedParameters(**self.base_parameters.__dict__)
            optimized = self._adjust_for_regime(optimized, market_conditions)
            optimized = self._adjust_for_volatility(optimized, market_conditions.volatility)
            optimized = self._adjust_for_time(optimized, market_conditions.time_of_day)
            if recent_performance:
                optimized = self._adjust_for_performance(optimized, recent_performance)
            optimized = self._enforce_bounds(optimized)
            self.current_parameters = optimized
            self.last_optimization = datetime.now(UTC)
            self.parameter_history.append({'timestamp': self.last_optimization, 'parameters': optimized, 'market_conditions': market_conditions, 'performance_trigger': recent_performance[-1] if recent_performance else 0.0})
            self.logger.info(f'Parameters optimized for {market_conditions.regime.value} regime')
            return optimized
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error optimizing parameters: {e}')
            return self.current_parameters

    def _should_optimize(self) -> bool:
        """Check if parameters should be optimized."""
        if not self.optimization_enabled:
            return False
        if self.last_optimization is None:
            return True
        time_since_last = datetime.now(UTC) - self.last_optimization
        return time_since_last >= self.optimization_frequency

    def _adjust_for_regime(self, params: OptimizedParameters, conditions: MarketConditions) -> OptimizedParameters:
        """Adjust parameters based on market regime."""
        if conditions.regime == MarketRegime.VOLATILE:
            params.position_size_multiplier *= 0.7
            params.stop_loss_multiplier *= 0.8
            params.rsi_oversold = 25.0
            params.rsi_overbought = 75.0
        elif conditions.regime == MarketRegime.LOW_VOLATILITY:
            params.position_size_multiplier *= 1.3
            params.take_profit_multiplier *= 1.2
        elif conditions.regime == MarketRegime.TRENDING:
            params.moving_average_period = 15
            params.take_profit_multiplier *= 1.5
        elif conditions.regime == MarketRegime.BULL_MARKET:
            params.position_size_multiplier *= 1.2
            params.rsi_oversold = 35.0
        elif conditions.regime == MarketRegime.BEAR_MARKET:
            params.position_size_multiplier *= 0.6
            params.stop_loss_multiplier *= 0.7
            params.rsi_overbought = 65.0
        return params

    def _adjust_for_volatility(self, params: OptimizedParameters, volatility: float) -> OptimizedParameters:
        """Adjust parameters based on volatility level."""
        vol_factor = min(2.0, max(0.5, volatility / 0.2))
        params.position_size_multiplier /= vol_factor
        params.stop_loss_multiplier *= vol_factor ** 0.5
        if volatility > 0.3:
            params.volatility_lookback = 10
        elif volatility < 0.15:
            params.volatility_lookback = 30
        return params

    def _adjust_for_time(self, params: OptimizedParameters, phase: TradingPhase) -> OptimizedParameters:
        """Adjust parameters based on trading phase."""
        if phase == TradingPhase.MARKET_OPEN:
            params.position_size_multiplier *= 0.8
            params.volume_threshold *= 1.5
        elif phase == TradingPhase.LATE_DAY:
            params.position_size_multiplier *= 0.9
            params.stop_loss_multiplier *= 0.9
        elif phase in [TradingPhase.PRE_MARKET, TradingPhase.AFTER_HOURS]:
            params.position_size_multiplier *= 0.5
            params.volume_threshold *= 2.0
        return params

    def _adjust_for_performance(self, params: OptimizedParameters, performance: list[float]) -> OptimizedParameters:
        """Adjust parameters based on recent performance."""
        if len(performance) < 5:
            return params
        recent_performance = performance[-10:]
        win_rate = sum((1 for p in recent_performance if p > 0)) / len(recent_performance)
        avg_return = statistics.mean(recent_performance)
        if win_rate < 0.4:
            params.position_size_multiplier *= 0.8
            params.stop_loss_multiplier *= 0.9
        elif win_rate > 0.7:
            params.position_size_multiplier *= 1.1
            params.take_profit_multiplier *= 1.1
        if avg_return < -0.02:
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

    def calculate_optimal_position_size(self, symbol: str, price: float, account_value: float, volatility: float, market_conditions: MarketConditions) -> float:
        """Calculate optimal position size using Kelly criterion and risk management with thread safety."""
        with self._lock:
            try:
                if price <= 0 or account_value <= 0:
                    self.logger.warning(f'Invalid inputs: price={price}, account_value={account_value}')
                    return 0
                kelly_fraction = self._calculate_kelly_fraction(symbol)
                base_position_pct = 0.02
                kelly_adjusted_pct = min(kelly_fraction, 0.25) if kelly_fraction > 0 else base_position_pct
                regime_multiplier = self._get_regime_multiplier(market_conditions.regime)
                epsilon = 1e-08
                volatility_adjustment = 1.0 / (1.0 + max(volatility, epsilon) * 5)
                time_multiplier = self._get_time_multiplier(market_conditions.time_of_day)
                param_multiplier = self.current_parameters.position_size_multiplier
                position_pct = kelly_adjusted_pct * regime_multiplier * volatility_adjustment * time_multiplier * param_multiplier
                position_value = account_value * position_pct
                shares = int(position_value / price)
                min_shares = max(1, int(account_value * 0.001 / price))
                max_shares = int(account_value * 0.1 / price)
                final_shares = max(min_shares, min(max_shares, shares))
                self.logger.debug(f'Position size calculated for {symbol}: {final_shares} shares (${final_shares * price:.2f}, {final_shares * price / account_value * 100:.2f}% of account) Kelly fraction: {kelly_fraction:.4f}')
                return final_shares
            except (ValueError, TypeError) as e:
                self.logger.error(f'Error calculating position size: {e}')
                return 0

    def _calculate_kelly_fraction(self, symbol: str) -> float:
        """Calculate Kelly fraction from historical performance with division by zero protection."""
        try:
            symbol_performance = []
            for record in self.performance_history:
                if isinstance(record, dict) and record.get('symbol') == symbol:
                    symbol_performance.append(record.get('return', 0.0))
            if len(symbol_performance) < 10:
                return 0.02
            wins = [p for p in symbol_performance if p > 0]
            losses = [abs(p) for p in symbol_performance if p < 0]
            total_trades = len(symbol_performance)
            if total_trades == 0:
                return 0.02
            win_rate = len(wins) / total_trades
            epsilon = 1e-08
            avg_win = sum(wins) / max(len(wins), 1) if wins else epsilon
            avg_loss = sum(losses) / max(len(losses), 1) if losses else epsilon
            if avg_loss <= epsilon:
                return min(0.25, win_rate * 0.5)
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            return kelly_fraction
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error calculating Kelly fraction: {e}')
            return 0.02
            return max(1, int(account_value * 0.001 / price))

    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier for market regime."""
        multipliers = {MarketRegime.BULL_MARKET: 1.2, MarketRegime.TRENDING: 1.1, MarketRegime.SIDEWAYS: 1.0, MarketRegime.LOW_VOLATILITY: 1.3, MarketRegime.VOLATILE: 0.7, MarketRegime.BEAR_MARKET: 0.6}
        return multipliers.get(regime, 1.0)

    def _get_time_multiplier(self, phase: TradingPhase) -> float:
        """Get position size multiplier for trading phase."""
        multipliers = {TradingPhase.PRE_MARKET: 0.5, TradingPhase.MARKET_OPEN: 0.8, TradingPhase.MID_DAY: 1.0, TradingPhase.LATE_DAY: 0.9, TradingPhase.AFTER_HOURS: 0.4}
        return multipliers.get(phase, 1.0)

    def calculate_stop_loss(self, entry_price: float, side: str, volatility: float, atr: float) -> float:
        """Calculate optimal stop loss price."""
        try:
            base_stop_pct = 0.02
            volatility_stop = volatility * 2
            atr_stop = atr * 2 / entry_price
            stop_pct = min(0.05, max(base_stop_pct, volatility_stop, atr_stop))
            stop_pct *= self.current_parameters.stop_loss_multiplier
            if side.upper() == 'BUY':
                stop_price = entry_price * (1 - stop_pct)
            else:
                stop_price = entry_price * (1 + stop_pct)
            return round(stop_price, 2)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error calculating stop loss: {e}')
            if side.upper() == 'BUY':
                return entry_price * 0.98
            else:
                return entry_price * 1.02

    def calculate_take_profit(self, entry_price: float, side: str, stop_loss: float, risk_reward_ratio: float=2.0) -> float:
        """Calculate optimal take profit price."""
        try:
            if side.upper() == 'BUY':
                risk_amount = entry_price - stop_loss
                take_profit_price = entry_price + risk_amount * risk_reward_ratio * self.current_parameters.take_profit_multiplier
            else:
                risk_amount = stop_loss - entry_price
                take_profit_price = entry_price - risk_amount * risk_reward_ratio * self.current_parameters.take_profit_multiplier
            return round(take_profit_price, 2)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Error calculating take profit: {e}')
            if side.upper() == 'BUY':
                return entry_price * 1.04
            else:
                return entry_price * 0.96

    def validate_mathematical_models(self) -> dict[str, Any]:
        """Validate mathematical models and calculations."""
        validation_results = {'timestamp': datetime.now(UTC).isoformat(), 'tests_run': [], 'tests_passed': 0, 'tests_failed': 0, 'issues_found': []}
        try:
            test_size = self.calculate_optimal_position_size('TEST', 100.0, 10000.0, 0.2, MarketConditions(MarketRegime.SIDEWAYS, 0.2, 0.5, 1.0, 0.5, 0.0, 20.0, TradingPhase.MID_DAY))
            assert test_size > 0, 'Position size must be positive'
            assert test_size <= 100, 'Position size should be reasonable for test account'
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('position_sizing')
        except (ValueError, TypeError) as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f'Position sizing error: {e}')
        try:
            stop = self.calculate_stop_loss(100.0, 'BUY', 0.02, 2.0)
            assert 90.0 <= stop <= 99.0, f'Stop loss {stop} should be reasonable'
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('stop_loss_calculation')
        except (ValueError, TypeError) as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f'Stop loss calculation error: {e}')
        try:
            profit = self.calculate_take_profit(100.0, 'BUY', 98.0, 2.0)
            assert profit > 100.0, f'Take profit {profit} should be above entry'
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('take_profit_calculation')
        except (ValueError, TypeError) as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f'Take profit calculation error: {e}')
        try:
            conditions = MarketConditions(MarketRegime.VOLATILE, 0.3, 0.8, 1.2, 0.6, 0.0, 25.0, TradingPhase.MID_DAY)
            optimized = self.optimize_parameters(conditions, [0.01, -0.02, 0.03], True)
            assert hasattr(optimized, 'position_size_multiplier'), 'Optimized parameters missing attributes'
            validation_results['tests_passed'] += 1
            validation_results['tests_run'].append('parameter_optimization')
        except (ValueError, TypeError) as e:
            validation_results['tests_failed'] += 1
            validation_results['issues_found'].append(f'Parameter optimization error: {e}')
        return validation_results

    def get_optimization_report(self) -> dict[str, Any]:
        """Get comprehensive optimization report."""
        return {'timestamp': datetime.now(UTC).isoformat(), 'optimization_enabled': self.optimization_enabled, 'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None, 'current_parameters': self.current_parameters.__dict__, 'base_parameters': self.base_parameters.__dict__, 'recent_regimes': [regime.regime.value for regime in list(self.market_regimes)[-10:]], 'parameter_changes': len(self.parameter_history), 'regime_performance': {regime.value: {'count': len(performances), 'avg_performance': statistics.mean(performances) if performances else 0.0, 'win_rate': sum((1 for p in performances if p > 0)) / len(performances) * 100 if performances else 0.0} for regime, performances in self.regime_performance.items()}}
_algorithm_optimizer: AlgorithmOptimizer | None = None

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