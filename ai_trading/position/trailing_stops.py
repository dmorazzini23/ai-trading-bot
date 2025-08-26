"""
Dynamic Trailing Stop Manager for intelligent position management.

Implements sophisticated trailing stop strategies:
- Volatility-adjusted trailing stops using ATR
- Momentum-based trailing (tighter when momentum weakens)
- Time-decay trailing (gradually tighten as position ages)
- Breakeven protection after specified gain thresholds

AI-AGENT-REF: Advanced trailing stop management with multiple algorithms
"""
from ai_trading.logging import get_logger
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import pandas as pd
logger = get_logger(__name__)

class TrailingStopType(Enum):
    """Types of trailing stop algorithms."""
    FIXED_PERCENT = 'fixed_percent'
    ATR_BASED = 'atr_based'
    MOMENTUM_BASED = 'momentum_based'
    TIME_DECAY = 'time_decay'
    BREAKEVEN_PROTECT = 'breakeven_protect'
    ADAPTIVE = 'adaptive'

@dataclass
class TrailingStopLevel:
    """Trailing stop level information."""
    symbol: str
    current_price: float
    stop_price: float
    stop_type: TrailingStopType
    trail_distance: float
    max_price_achieved: float
    entry_price: float
    unrealized_gain_pct: float
    days_held: int
    last_updated: datetime
    is_triggered: bool = False
    trigger_reason: str = ''

class TrailingStopManager:
    """
    Manage dynamic trailing stops for position protection.

    Implements multiple trailing stop algorithms:
    - Volatility-adjusted using ATR
    - Momentum-based adjustments
    - Time-decay mechanisms
    - Breakeven protection
    - Adaptive combination of methods
    """

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.logger = get_logger(__name__ + '.TrailingStopManager')
        self.base_trail_percent = 3.0
        self.atr_multiplier = 2.0
        self.atr_period = 14
        self.momentum_period = 14
        self.strong_momentum_threshold = 0.7
        self.weak_momentum_threshold = 0.3
        self.time_decay_start_days = 7
        self.max_time_decay = 0.5
        self.breakeven_trigger = 1.5
        self.breakeven_buffer = 0.1
        self.stop_levels: dict[str, TrailingStopLevel] = {}

    def update_trailing_stop(self, symbol: str, position_data: Any, current_price: float) -> TrailingStopLevel | None:
        """
        Update trailing stop for a position.

        Args:
            symbol: Symbol to update
            position_data: Current position information
            current_price: Current market price

        Returns:
            Updated TrailingStopLevel or None if no position
        """
        try:
            if not position_data:
                return None
            entry_price = float(getattr(position_data, 'avg_entry_price', 0))
            qty = int(getattr(position_data, 'qty', 0))
            if entry_price <= 0 or qty == 0:
                return None
            unrealized_gain_pct = (current_price - entry_price) / entry_price * 100
            if symbol not in self.stop_levels:
                self.stop_levels[symbol] = self._initialize_stop_level(symbol, entry_price, current_price, position_data)
            stop_level = self.stop_levels[symbol]
            stop_level.max_price_achieved = max(stop_level.max_price_achieved, current_price)
            stop_level.current_price = current_price
            stop_level.unrealized_gain_pct = unrealized_gain_pct
            stop_level.days_held = self._calculate_days_held(position_data)
            stop_level.last_updated = datetime.now(UTC)
            new_stop_price = self._calculate_adaptive_stop(symbol, stop_level)
            if qty > 0:
                stop_level.stop_price = max(stop_level.stop_price, new_stop_price)
            else:
                stop_level.stop_price = min(stop_level.stop_price, new_stop_price)
            stop_level.trail_distance = abs((current_price - stop_level.stop_price) / current_price) * 100
            self._check_stop_trigger(stop_level, qty)
            self.logger.info('TRAILING_STOP_UPDATE | %s price=%.2f stop=%.2f trail=%.2f%% gain=%.2f%%', symbol, current_price, stop_level.stop_price, stop_level.trail_distance, unrealized_gain_pct)
            return stop_level
        except (ValueError, TypeError) as exc:
            self.logger.warning('update_trailing_stop failed for %s: %s', symbol, exc)
            return None

    def get_stop_level(self, symbol: str) -> TrailingStopLevel | None:
        """Get current trailing stop level for symbol."""
        return self.stop_levels.get(symbol)

    def remove_stop_level(self, symbol: str) -> None:
        """Remove trailing stop tracking for symbol."""
        if symbol in self.stop_levels:
            self.logger.info('TRAILING_STOP_REMOVED | %s', symbol)
            del self.stop_levels[symbol]

    def get_triggered_stops(self) -> list[TrailingStopLevel]:
        """Get list of triggered trailing stops that need action."""
        return [stop for stop in self.stop_levels.values() if stop.is_triggered]

    def _initialize_stop_level(self, symbol: str, entry_price: float, current_price: float, position_data: Any) -> TrailingStopLevel:
        """Initialize new trailing stop level."""
        initial_distance = self._calculate_initial_stop_distance(symbol)
        qty = int(getattr(position_data, 'qty', 0))
        if qty > 0:
            initial_stop = current_price * (1 - initial_distance / 100)
        else:
            initial_stop = current_price * (1 + initial_distance / 100)
        stop_level = TrailingStopLevel(symbol=symbol, current_price=current_price, stop_price=initial_stop, stop_type=TrailingStopType.ADAPTIVE, trail_distance=initial_distance, max_price_achieved=current_price, entry_price=entry_price, unrealized_gain_pct=(current_price - entry_price) / entry_price * 100, days_held=0, last_updated=datetime.now(UTC))
        self.logger.info('TRAILING_STOP_INIT | %s entry=%.2f current=%.2f stop=%.2f distance=%.2f%%', symbol, entry_price, current_price, initial_stop, initial_distance)
        return stop_level

    def _calculate_adaptive_stop(self, symbol: str, stop_level: TrailingStopLevel) -> float:
        """Calculate adaptive trailing stop using multiple algorithms."""
        try:
            current_price = stop_level.current_price
            entry_price = stop_level.entry_price
            market_data = self._get_market_data(symbol)
            distances = {}
            distances['fixed'] = self.base_trail_percent
            if market_data is not None:
                atr_distance = self._calculate_atr_stop_distance(market_data)
                distances['atr'] = atr_distance
            else:
                distances['atr'] = self.base_trail_percent
            momentum_multiplier = self._calculate_momentum_multiplier(symbol, market_data)
            distances['momentum'] = self.base_trail_percent * momentum_multiplier
            time_multiplier = self._calculate_time_decay_multiplier(stop_level.days_held)
            distances['time_decay'] = self.base_trail_percent * time_multiplier
            breakeven_distance = self._calculate_breakeven_distance(stop_level)
            if breakeven_distance is not None:
                distances['breakeven'] = breakeven_distance
            final_distance = self._combine_stop_distances(distances, stop_level)
            if hasattr(stop_level, 'qty') or current_price > entry_price:
                new_stop = current_price * (1 - final_distance / 100)
            else:
                new_stop = current_price * (1 + final_distance / 100)
            return new_stop
        except (ValueError, TypeError) as exc:
            self.logger.warning('_calculate_adaptive_stop failed for %s: %s', symbol, exc)
            return stop_level.current_price * (1 - self.base_trail_percent / 100)

    def _calculate_initial_stop_distance(self, symbol: str) -> float:
        """Calculate initial stop distance based on volatility."""
        try:
            market_data = self._get_market_data(symbol)
            if market_data is not None:
                atr_distance = self._calculate_atr_stop_distance(market_data)
                return max(self.base_trail_percent, atr_distance)
            return self.base_trail_percent
        except (KeyError, ValueError, TypeError, IndexError):
            return self.base_trail_percent

    def _calculate_atr_stop_distance(self, data: pd.DataFrame) -> float:
        """Calculate ATR-based stop distance."""
        try:
            if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns or (len(data) < self.atr_period):
                return self.base_trail_percent
            high = data['high']
            low = data['low']
            close = data['close']
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean()
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
            current_price = close.iloc[-1]
            if current_price > 0 and current_atr > 0:
                atr_percent = current_atr * self.atr_multiplier / current_price * 100
                return max(1.0, min(10.0, atr_percent))
            return self.base_trail_percent
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
            return self.base_trail_percent

    def _calculate_momentum_multiplier(self, symbol: str, data: pd.DataFrame | None) -> float:
        """Calculate momentum-based multiplier for stop distance."""
        try:
            if data is None or 'close' not in data.columns or len(data) < self.momentum_period:
                return 1.0
            closes = data['close']
            rsi = self._calculate_rsi(closes, self.momentum_period)
            if pd.isna(rsi):
                return 1.0
            momentum_score = rsi / 100.0
            if momentum_score > self.strong_momentum_threshold:
                return 1.3
            elif momentum_score < self.weak_momentum_threshold:
                return 0.7
            else:
                return 1.0
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return 1.0

    def _calculate_time_decay_multiplier(self, days_held: int) -> float:
        """Calculate time-based multiplier for gradually tightening stops."""
        try:
            if days_held <= self.time_decay_start_days:
                return 1.0
            decay_days = days_held - self.time_decay_start_days
            max_decay_days = 30
            decay_factor = min(1.0, decay_days / max_decay_days)
            multiplier = 1.0 - decay_factor * self.max_time_decay
            return max(0.5, multiplier)
        except (ValueError, TypeError, ZeroDivisionError):
            return 1.0

    def _calculate_breakeven_distance(self, stop_level: TrailingStopLevel) -> float | None:
        """Calculate breakeven protection distance."""
        try:
            if stop_level.unrealized_gain_pct >= self.breakeven_trigger:
                entry_price = stop_level.entry_price
                current_price = stop_level.current_price
                breakeven_price = entry_price * (1 + self.breakeven_buffer / 100)
                distance = (current_price - breakeven_price) / current_price * 100
                return max(0.1, distance)
            return None
        except (KeyError, ValueError, TypeError):
            return None

    def _combine_stop_distances(self, distances: dict[str, float], stop_level: TrailingStopLevel) -> float:
        """Combine multiple stop distance calculations."""
        try:
            weights = {'fixed': 0.2, 'atr': 0.3, 'momentum': 0.2, 'time_decay': 0.2, 'breakeven': 0.1}
            if 'breakeven' in distances:
                weights['breakeven'] = 0.4
                weights['atr'] = 0.2
                weights['momentum'] = 0.2
                weights['time_decay'] = 0.1
                weights['fixed'] = 0.1
            total_weight = 0
            weighted_sum = 0
            for method, distance in distances.items():
                if method in weights:
                    weight = weights[method]
                    weighted_sum += distance * weight
                    total_weight += weight
            if total_weight > 0:
                final_distance = weighted_sum / total_weight
            else:
                final_distance = self.base_trail_percent
            final_distance = max(0.5, min(15.0, final_distance))
            return final_distance
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return self.base_trail_percent

    def _check_stop_trigger(self, stop_level: TrailingStopLevel, qty: int) -> None:
        """Check if trailing stop has been triggered."""
        try:
            current_price = stop_level.current_price
            stop_price = stop_level.stop_price
            if qty > 0:
                if current_price <= stop_price:
                    stop_level.is_triggered = True
                    stop_level.trigger_reason = f'Price {current_price:.2f} <= Stop {stop_price:.2f}'
            elif current_price >= stop_price:
                stop_level.is_triggered = True
                stop_level.trigger_reason = f'Price {current_price:.2f} >= Stop {stop_price:.2f}'
            if stop_level.is_triggered:
                self.logger.warning('TRAILING_STOP_TRIGGERED | %s %s', stop_level.symbol, stop_level.trigger_reason)
        except (ValueError, TypeError) as exc:
            self.logger.warning('_check_stop_trigger failed: %s', exc)

    def _get_market_data(self, symbol: str) -> pd.DataFrame | None:
        """Get market data for stop calculations."""
        try:
            if self.ctx and hasattr(self.ctx, 'data_fetcher'):
                df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
                if df is not None and (not df.empty) and (len(df) >= self.atr_period):
                    return df
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and (not df.empty):
                    return df
            return None
        except (AttributeError, ValueError, KeyError):
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int=14) -> float:
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
                return 50.0
            deltas = prices.diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
            return 50.0

    def _calculate_days_held(self, position_data: Any) -> int:
        """Calculate number of days position has been held."""
        try:
            return 0
        except (AttributeError, ValueError, TypeError):
            return 0