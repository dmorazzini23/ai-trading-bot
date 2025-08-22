"""
Dynamic Trailing Stop Manager for intelligent position management.

Implements sophisticated trailing stop strategies:
- Volatility-adjusted trailing stops using ATR
- Momentum-based trailing (tighter when momentum weakens)
- Time-decay trailing (gradually tighten as position ages)
- Breakeven protection after specified gain thresholds

AI-AGENT-REF: Advanced trailing stop management with multiple algorithms
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# AI-AGENT-REF: graceful imports with fallbacks
# Use hard imports since numpy and pandas are dependencies
import pandas as pd

logger = logging.getLogger(__name__)


class TrailingStopType(Enum):
    """Types of trailing stop algorithms."""

    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    MOMENTUM_BASED = "momentum_based"
    TIME_DECAY = "time_decay"
    BREAKEVEN_PROTECT = "breakeven_protect"
    ADAPTIVE = "adaptive"


@dataclass
class TrailingStopLevel:
    """Trailing stop level information."""

    symbol: str
    current_price: float
    stop_price: float
    stop_type: TrailingStopType
    trail_distance: float  # Current trailing distance in %
    max_price_achieved: float  # Highest price since position entry
    entry_price: float
    unrealized_gain_pct: float
    days_held: int
    last_updated: datetime
    is_triggered: bool = False
    trigger_reason: str = ""


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
        self.logger = logging.getLogger(__name__ + ".TrailingStopManager")

        # Trailing stop parameters
        self.base_trail_percent = 3.0  # Base trailing distance
        self.atr_multiplier = 2.0  # ATR multiplier for volatility-based stops
        self.atr_period = 14  # ATR calculation period

        # Momentum adjustment parameters
        self.momentum_period = 14
        self.strong_momentum_threshold = 0.7
        self.weak_momentum_threshold = 0.3

        # Time decay parameters
        self.time_decay_start_days = 7  # Start tightening after 7 days
        self.max_time_decay = 0.5  # Maximum tightening factor

        # Breakeven protection
        self.breakeven_trigger = 1.5  # Move to breakeven after 1.5% gain
        self.breakeven_buffer = 0.1  # Small buffer above breakeven

        # Position tracking
        self.stop_levels: dict[str, TrailingStopLevel] = {}

    def update_trailing_stop(
        self, symbol: str, position_data: Any, current_price: float
    ) -> TrailingStopLevel | None:
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

            # Get position details
            entry_price = float(getattr(position_data, "avg_entry_price", 0))
            qty = int(getattr(position_data, "qty", 0))

            if entry_price <= 0 or qty == 0:
                return None

            # Calculate current metrics
            unrealized_gain_pct = ((current_price - entry_price) / entry_price) * 100

            # Get or create stop level
            if symbol not in self.stop_levels:
                self.stop_levels[symbol] = self._initialize_stop_level(
                    symbol, entry_price, current_price, position_data
                )

            stop_level = self.stop_levels[symbol]

            # Update max price achieved
            stop_level.max_price_achieved = max(
                stop_level.max_price_achieved, current_price
            )
            stop_level.current_price = current_price
            stop_level.unrealized_gain_pct = unrealized_gain_pct
            stop_level.days_held = self._calculate_days_held(position_data)
            stop_level.last_updated = datetime.now(UTC)

            # Calculate new stop price using multiple methods
            new_stop_price = self._calculate_adaptive_stop(symbol, stop_level)

            # Only move stop up (for long positions)
            if qty > 0:  # Long position
                stop_level.stop_price = max(stop_level.stop_price, new_stop_price)
            else:  # Short position
                stop_level.stop_price = min(stop_level.stop_price, new_stop_price)

            # Update trail distance
            stop_level.trail_distance = (
                abs((current_price - stop_level.stop_price) / current_price) * 100
            )

            # Check if stop is triggered
            self._check_stop_trigger(stop_level, qty)

            self.logger.info(
                "TRAILING_STOP_UPDATE | %s price=%.2f stop=%.2f trail=%.2f%% gain=%.2f%%",
                symbol,
                current_price,
                stop_level.stop_price,
                stop_level.trail_distance,
                unrealized_gain_pct,
            )

            return stop_level

        except Exception as exc:
            self.logger.warning("update_trailing_stop failed for %s: %s", symbol, exc)
            return None

    def get_stop_level(self, symbol: str) -> TrailingStopLevel | None:
        """Get current trailing stop level for symbol."""
        return self.stop_levels.get(symbol)

    def remove_stop_level(self, symbol: str) -> None:
        """Remove trailing stop tracking for symbol."""
        if symbol in self.stop_levels:
            self.logger.info("TRAILING_STOP_REMOVED | %s", symbol)
            del self.stop_levels[symbol]

    def get_triggered_stops(self) -> list[TrailingStopLevel]:
        """Get list of triggered trailing stops that need action."""
        return [stop for stop in self.stop_levels.values() if stop.is_triggered]

    def _initialize_stop_level(
        self, symbol: str, entry_price: float, current_price: float, position_data: Any
    ) -> TrailingStopLevel:
        """Initialize new trailing stop level."""

        # Calculate initial stop distance
        initial_distance = self._calculate_initial_stop_distance(symbol)

        # Set initial stop price
        qty = int(getattr(position_data, "qty", 0))
        if qty > 0:  # Long position
            initial_stop = current_price * (1 - initial_distance / 100)
        else:  # Short position
            initial_stop = current_price * (1 + initial_distance / 100)

        stop_level = TrailingStopLevel(
            symbol=symbol,
            current_price=current_price,
            stop_price=initial_stop,
            stop_type=TrailingStopType.ADAPTIVE,
            trail_distance=initial_distance,
            max_price_achieved=current_price,
            entry_price=entry_price,
            unrealized_gain_pct=((current_price - entry_price) / entry_price) * 100,
            days_held=0,
            last_updated=datetime.now(UTC),
        )

        self.logger.info(
            "TRAILING_STOP_INIT | %s entry=%.2f current=%.2f stop=%.2f distance=%.2f%%",
            symbol,
            entry_price,
            current_price,
            initial_stop,
            initial_distance,
        )

        return stop_level

    def _calculate_adaptive_stop(
        self, symbol: str, stop_level: TrailingStopLevel
    ) -> float:
        """Calculate adaptive trailing stop using multiple algorithms."""
        try:
            current_price = stop_level.current_price
            entry_price = stop_level.entry_price

            # Get market data for volatility and momentum
            market_data = self._get_market_data(symbol)

            # Calculate stop distances using different methods
            distances = {}

            # 1. Fixed percentage
            distances["fixed"] = self.base_trail_percent

            # 2. ATR-based (volatility-adjusted)
            if market_data is not None:
                atr_distance = self._calculate_atr_stop_distance(market_data)
                distances["atr"] = atr_distance
            else:
                distances["atr"] = self.base_trail_percent

            # 3. Momentum-based adjustment
            momentum_multiplier = self._calculate_momentum_multiplier(
                symbol, market_data
            )
            distances["momentum"] = self.base_trail_percent * momentum_multiplier

            # 4. Time-decay adjustment
            time_multiplier = self._calculate_time_decay_multiplier(
                stop_level.days_held
            )
            distances["time_decay"] = self.base_trail_percent * time_multiplier

            # 5. Breakeven protection
            breakeven_distance = self._calculate_breakeven_distance(stop_level)
            if breakeven_distance is not None:
                distances["breakeven"] = breakeven_distance

            # Combine distances using weighted average
            final_distance = self._combine_stop_distances(distances, stop_level)

            # Calculate new stop price
            if (
                hasattr(stop_level, "qty") or current_price > entry_price
            ):  # Assume long for now
                new_stop = current_price * (1 - final_distance / 100)
            else:  # Short position
                new_stop = current_price * (1 + final_distance / 100)

            return new_stop

        except Exception as exc:
            self.logger.warning(
                "_calculate_adaptive_stop failed for %s: %s", symbol, exc
            )
            # Fallback to simple percentage
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
            if (
                "high" not in data.columns
                or "low" not in data.columns
                or "close" not in data.columns
                or len(data) < self.atr_period
            ):
                return self.base_trail_percent

            # Calculate True Range
            high = data["high"]
            low = data["low"]
            close = data["close"]
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR
            atr = true_range.rolling(window=self.atr_period).mean()
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

            # Convert ATR to percentage of current price
            current_price = close.iloc[-1]
            if current_price > 0 and current_atr > 0:
                atr_percent = (current_atr * self.atr_multiplier / current_price) * 100
                return max(1.0, min(10.0, atr_percent))  # Cap between 1% and 10%

            return self.base_trail_percent

        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
            return self.base_trail_percent

    def _calculate_momentum_multiplier(
        self, symbol: str, data: pd.DataFrame | None
    ) -> float:
        """Calculate momentum-based multiplier for stop distance."""
        try:
            if (
                data is None
                or "close" not in data.columns
                or len(data) < self.momentum_period
            ):
                return 1.0

            # Calculate RSI for momentum
            closes = data["close"]
            rsi = self._calculate_rsi(closes, self.momentum_period)

            if pd.isna(rsi):
                return 1.0

            # Convert RSI to momentum score (0-1)
            momentum_score = rsi / 100.0

            # Adjust multiplier based on momentum
            if momentum_score > self.strong_momentum_threshold:
                # Strong momentum - wider stops (let winners run)
                return 1.3
            elif momentum_score < self.weak_momentum_threshold:
                # Weak momentum - tighter stops (protect capital)
                return 0.7
            else:
                # Neutral momentum
                return 1.0

        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return 1.0

    def _calculate_time_decay_multiplier(self, days_held: int) -> float:
        """Calculate time-based multiplier for gradually tightening stops."""
        try:
            if days_held <= self.time_decay_start_days:
                return 1.0

            # Gradually tighten stops over time
            decay_days = days_held - self.time_decay_start_days
            max_decay_days = 30  # Full decay after 30 additional days

            decay_factor = min(1.0, decay_days / max_decay_days)
            multiplier = 1.0 - (decay_factor * self.max_time_decay)

            return max(0.5, multiplier)  # Never go below 50% of base distance

        except (ValueError, TypeError, ZeroDivisionError):
            return 1.0

    def _calculate_breakeven_distance(
        self, stop_level: TrailingStopLevel
    ) -> float | None:
        """Calculate breakeven protection distance."""
        try:
            if stop_level.unrealized_gain_pct >= self.breakeven_trigger:
                # Move stop to just above breakeven
                entry_price = stop_level.entry_price
                current_price = stop_level.current_price

                # Calculate distance to breakeven + buffer
                breakeven_price = entry_price * (1 + self.breakeven_buffer / 100)
                distance = ((current_price - breakeven_price) / current_price) * 100

                return max(0.1, distance)  # Minimum 0.1% distance

            return None

        except (KeyError, ValueError, TypeError):
            return None

    def _combine_stop_distances(
        self, distances: dict[str, float], stop_level: TrailingStopLevel
    ) -> float:
        """Combine multiple stop distance calculations."""
        try:
            # Weights for different methods
            weights = {
                "fixed": 0.2,
                "atr": 0.3,
                "momentum": 0.2,
                "time_decay": 0.2,
                "breakeven": 0.1,
            }

            # If breakeven protection is active, prioritize it
            if "breakeven" in distances:
                weights["breakeven"] = 0.4
                weights["atr"] = 0.2
                weights["momentum"] = 0.2
                weights["time_decay"] = 0.1
                weights["fixed"] = 0.1

            # Calculate weighted average
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

            # Sanity checks
            final_distance = max(0.5, min(15.0, final_distance))

            return final_distance

        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return self.base_trail_percent

    def _check_stop_trigger(self, stop_level: TrailingStopLevel, qty: int) -> None:
        """Check if trailing stop has been triggered."""
        try:
            current_price = stop_level.current_price
            stop_price = stop_level.stop_price

            if qty > 0:  # Long position
                if current_price <= stop_price:
                    stop_level.is_triggered = True
                    stop_level.trigger_reason = (
                        f"Price {current_price:.2f} <= Stop {stop_price:.2f}"
                    )
            elif current_price >= stop_price:
                stop_level.is_triggered = True
                stop_level.trigger_reason = (
                    f"Price {current_price:.2f} >= Stop {stop_price:.2f}"
                )

            if stop_level.is_triggered:
                self.logger.warning(
                    "TRAILING_STOP_TRIGGERED | %s %s",
                    stop_level.symbol,
                    stop_level.trigger_reason,
                )

        except Exception as exc:
            self.logger.warning("_check_stop_trigger failed: %s", exc)

    def _get_market_data(self, symbol: str) -> pd.DataFrame | None:
        """Get market data for stop calculations."""
        try:
            if self.ctx and hasattr(self.ctx, "data_fetcher"):
                # Try minute data first
                df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
                if df is not None and not df.empty and len(df) >= self.atr_period:
                    return df

                # Fallback to daily data
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and not df.empty:
                    return df

            return None

        except (AttributeError, ValueError, KeyError):
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
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
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
            return 50.0

    def _calculate_days_held(self, position_data: Any) -> int:
        """Calculate number of days position has been held."""
        try:
            # This would need to be implemented based on position_data structure
            # For now, return a default value
            return 0

        except (AttributeError, ValueError, TypeError):
            return 0
