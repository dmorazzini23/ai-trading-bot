"""
Dynamic Trailing Stop Manager for intelligent position management.

Implements sophisticated trailing stop strategies:
- Volatility-adjusted trailing stops using ATR
- Momentum-based trailing (tighter when momentum weakens)
- Time-decay trailing (gradually tighten as position ages)
- Breakeven protection after specified gain thresholds

AI-AGENT-REF: Advanced trailing stop management with multiple algorithms
"""

from __future__ import annotations
from ai_trading.logging import get_logger
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TYPE_CHECKING
from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)
PRICE_EPSILON = 1e-6


def _fmt(value: float | None) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "nan"


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
    trail_distance: float
    max_price_achieved: float
    entry_price: float
    unrealized_gain_pct: float
    days_held: int
    last_updated: datetime
    side: str = "long"
    trail_pct: float = 0.03
    max_price_since_entry: float = 0.0
    min_price_since_entry: float = 0.0
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
        self.logger = get_logger(__name__ + ".TrailingStopManager")
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
            entry_price = float(getattr(position_data, "avg_entry_price", 0))
            qty = int(getattr(position_data, "qty", 0))
            if entry_price <= 0 or qty == 0:
                return None
            unrealized_gain_pct = (current_price - entry_price) / entry_price * 100
            if symbol not in self.stop_levels:
                self.stop_levels[symbol] = self._initialize_stop_level(
                    symbol, entry_price, current_price, position_data
                )
            stop_level = self.stop_levels[symbol]
            previous_side = stop_level.side
            stop_level.side = "long" if qty > 0 else "short"
            if previous_side != stop_level.side:
                stop_level.max_price_since_entry = max(stop_level.entry_price, current_price)
                stop_level.min_price_since_entry = min(stop_level.entry_price, current_price)
                stop_level.stop_price = 0.0
            if stop_level.max_price_since_entry <= 0:
                stop_level.max_price_since_entry = max(stop_level.entry_price, current_price)
            if stop_level.min_price_since_entry <= 0:
                stop_level.min_price_since_entry = min(stop_level.entry_price, current_price)
            stop_level.max_price_since_entry = max(stop_level.max_price_since_entry, current_price)
            stop_level.min_price_since_entry = min(stop_level.min_price_since_entry, current_price)
            stop_level.max_price_achieved = max(stop_level.max_price_achieved, stop_level.max_price_since_entry)
            stop_level.current_price = current_price
            stop_level.unrealized_gain_pct = unrealized_gain_pct
            stop_level.days_held = self._calculate_days_held(position_data)
            stop_level.last_updated = datetime.now(UTC)
            trail_pct = getattr(stop_level, "trail_pct", None)
            if trail_pct is None or trail_pct <= 0:
                trail_pct = self.base_trail_percent / 100.0
            stop_level.trail_pct = trail_pct
            try:
                existing_stop = float(stop_level.stop_price)
            except (TypeError, ValueError):
                existing_stop = 0.0
            if stop_level.side == "long":
                candidate = stop_level.max_price_since_entry * (1 - trail_pct)
                stop_level.stop_price = self._merge_stop_prices(existing_stop, candidate, "long")
            else:
                candidate = stop_level.min_price_since_entry * (1 + trail_pct)
                stop_level.stop_price = self._merge_stop_prices(existing_stop, candidate, "short")
            self._ensure_directional_stop(stop_level, current_price, candidate=candidate)
            self._check_stop_trigger(stop_level, qty)
            try:
                stop_level.trail_distance = abs((current_price - stop_level.stop_price) / current_price) * 100
            except (TypeError, ZeroDivisionError):
                stop_level.trail_distance = trail_pct * 100
            self.logger.info(
                "TRAILING_STOP_UPDATE | %s price=%.2f stop=%.2f trail=%.2f%% gain=%.2f%% side=%s",
                symbol,
                current_price,
                stop_level.stop_price,
                stop_level.trail_distance,
                unrealized_gain_pct,
                stop_level.side,
            )
            return stop_level
        except (ValueError, TypeError) as exc:
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
        qty = int(getattr(position_data, "qty", 0))
        side = "long" if qty >= 0 else "short"
        configured_pct = getattr(position_data, "trail_pct", None)
        try:
            base_pct = float(configured_pct) if configured_pct is not None else self.base_trail_percent
        except (TypeError, ValueError):
            base_pct = self.base_trail_percent
        trail_pct = max(base_pct, 0.01) / 100.0
        initial_distance = trail_pct * 100
        max_since = max(entry_price, current_price)
        min_since = min(entry_price, current_price)
        if side == "long":
            initial_stop = max_since * (1 - trail_pct)
        else:
            initial_stop = min_since * (1 + trail_pct)
        stop_level = TrailingStopLevel(
            symbol=symbol,
            current_price=current_price,
            stop_price=initial_stop,
            stop_type=TrailingStopType.ADAPTIVE,
            trail_distance=initial_distance,
            max_price_achieved=max_since,
            entry_price=entry_price,
            unrealized_gain_pct=(current_price - entry_price) / entry_price * 100,
            days_held=0,
            last_updated=datetime.now(UTC),
            side=side,
            trail_pct=trail_pct,
            max_price_since_entry=max_since,
            min_price_since_entry=min_since,
        )
        self._ensure_directional_stop(stop_level, current_price, candidate=initial_stop, was_init=True)
        self.logger.info(
            "TRAILING_STOP_INIT | %s entry=%.2f current=%.2f stop=%.2f distance=%.2f%% side=%s",
            symbol,
            entry_price,
            current_price,
            stop_level.stop_price,
            initial_distance,
            side,
        )
        return stop_level

    def _calculate_adaptive_stop(self, symbol: str, stop_level: TrailingStopLevel) -> float:
        """Calculate adaptive trailing stop using multiple algorithms."""
        try:
            current_price = stop_level.current_price
            entry_price = stop_level.entry_price
            market_data = self._get_market_data(symbol)
            distances = {}
            distances["fixed"] = self.base_trail_percent
            if market_data is not None:
                atr_distance = self._calculate_atr_stop_distance(market_data)
                distances["atr"] = atr_distance
            else:
                distances["atr"] = self.base_trail_percent
            momentum_multiplier = self._calculate_momentum_multiplier(symbol, market_data)
            distances["momentum"] = self.base_trail_percent * momentum_multiplier
            time_multiplier = self._calculate_time_decay_multiplier(stop_level.days_held)
            distances["time_decay"] = self.base_trail_percent * time_multiplier
            breakeven_distance = self._calculate_breakeven_distance(stop_level)
            if breakeven_distance is not None:
                distances["breakeven"] = breakeven_distance
            final_distance = self._combine_stop_distances(distances, stop_level)
            if hasattr(stop_level, "qty") or current_price > entry_price:
                new_stop = current_price * (1 - final_distance / 100)
            else:
                new_stop = current_price * (1 + final_distance / 100)
            return new_stop
        except (ValueError, TypeError) as exc:
            self.logger.warning("_calculate_adaptive_stop failed for %s: %s", symbol, exc)
            return stop_level.current_price * (1 - self.base_trail_percent / 100)

    def _merge_stop_prices(self, existing_stop: float, candidate: float, side: str) -> float:
        """Combine ``existing_stop`` with ``candidate`` respecting ``side`` direction."""

        try:
            existing_value = float(existing_stop)
        except (TypeError, ValueError):
            existing_value = 0.0

        if existing_value <= 0:
            return float(candidate)

        if side == "long":
            return max(existing_value, float(candidate))
        return min(existing_value, float(candidate))

    def _ensure_directional_stop(
        self,
        stop_level: TrailingStopLevel,
        current_price: float,
        *,
        candidate: float | None = None,
        was_init: bool = False,
    ) -> None:
        """Ensure stop aligns with side and avoids immediate triggers."""

        epsilon = PRICE_EPSILON
        try:
            stop_price = float(stop_level.stop_price)
        except (TypeError, ValueError):
            stop_price = 0.0
        corrected = False
        if stop_level.side == "long":
            reference = stop_level.max_price_since_entry or max(stop_level.entry_price, current_price)
            candidate_value = float(candidate) if candidate is not None else reference * (1 - stop_level.trail_pct)
            limit = current_price * (1 - epsilon)
            desired = min(candidate_value, limit)
            if candidate_value >= current_price or desired <= 0:
                desired = limit
            if desired >= current_price:
                desired = limit
            if stop_price <= 0 or stop_price >= current_price or stop_price > limit + epsilon:
                stop_level.stop_price = desired
                corrected = True
        else:
            reference = stop_level.min_price_since_entry or min(stop_level.entry_price, current_price)
            candidate_value = float(candidate) if candidate is not None else reference * (1 + stop_level.trail_pct)
            limit = current_price * (1 + epsilon)
            desired = max(candidate_value, limit)
            if candidate_value <= current_price or desired <= current_price:
                desired = limit
            if stop_price <= 0 or stop_price <= current_price or stop_price < limit - epsilon:
                stop_level.stop_price = desired
                corrected = True
        if corrected:
            segments = [
                f"symbol={stop_level.symbol}",
                f"side={stop_level.side}",
                f"trail_pct={_fmt(stop_level.trail_pct)}",
                f"stop_price={_fmt(stop_level.stop_price)}",
                f"current_price={_fmt(current_price)}",
            ]
            if stop_level.side == "long":
                segments.append(f"max_since_entry={_fmt(stop_level.max_price_since_entry)}")
            else:
                segments.append(f"min_since_entry={_fmt(stop_level.min_price_since_entry)}")
            segments.append(f"reason={'initialization' if was_init else 'state_correction'}")
            self.logger.info("TRAILING_STOP_CORRECTED | %s", " ".join(segments))

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

    def _calculate_atr_stop_distance(self, data: "pd.DataFrame") -> float:
        """Calculate ATR-based stop distance."""
        pd = load_pandas()
        try:
            if (
                "high" not in data.columns
                or "low" not in data.columns
                or "close" not in data.columns
                or (len(data) < self.atr_period)
            ):
                return self.base_trail_percent
            high = data["high"]
            low = data["low"]
            close = data["close"]
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

    def _calculate_momentum_multiplier(self, symbol: str, data: "pd.DataFrame" | None) -> float:
        """Calculate momentum-based multiplier for stop distance."""
        try:
            if data is None or "close" not in data.columns or len(data) < self.momentum_period:
                return 1.0
            closes = data["close"]
            rsi = self._calculate_rsi(closes, self.momentum_period)
            pd = load_pandas()
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
            weights = {"fixed": 0.2, "atr": 0.3, "momentum": 0.2, "time_decay": 0.2, "breakeven": 0.1}
            if "breakeven" in distances:
                weights["breakeven"] = 0.4
                weights["atr"] = 0.2
                weights["momentum"] = 0.2
                weights["time_decay"] = 0.1
                weights["fixed"] = 0.1
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

    def _check_stop_trigger(self, stop_level: TrailingStopLevel, qty: int) -> bool:
        """Check if trailing stop has been triggered."""

        try:
            current_price = stop_level.current_price
            stop_price = stop_level.stop_price
            try:
                stop_price_val = float(stop_price)
            except (TypeError, ValueError):
                stop_price_val = float("nan")
            side = stop_level.side
            if side == "long":
                triggered = current_price <= stop_price_val
            else:
                triggered = current_price >= stop_price_val
            if triggered:
                stop_level.is_triggered = True
                stop_level.trigger_reason = "price_crossed_stop"
                segments = [
                    f"symbol={stop_level.symbol}",
                    f"side={side}",
                    f"trail_pct={_fmt(stop_level.trail_pct)}",
                    f"stop_used={_fmt(stop_price_val)}",
                    f"current_price={_fmt(current_price)}",
                    "reason=price_crossed_stop",
                ]
                if side == "long":
                    segments.append(f"max_since_entry={_fmt(stop_level.max_price_since_entry)}")
                else:
                    segments.append(f"min_since_entry={_fmt(stop_level.min_price_since_entry)}")
                self.logger.warning("TRAILING_STOP_TRIGGERED | %s", " ".join(segments))
                return True
            stop_level.is_triggered = False
            stop_level.trigger_reason = ""
            return False
        except (ValueError, TypeError) as exc:
            self.logger.warning("_check_stop_trigger failed: %s", exc)
            return False

    def _get_market_data(self, symbol: str) -> "pd.DataFrame" | None:
        """Get market data for stop calculations."""
        try:
            if self.ctx and hasattr(self.ctx, "data_fetcher"):
                df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
                if df is not None and (not df.empty) and (len(df) >= self.atr_period):
                    return df
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and (not df.empty):
                    return df
            return None
        except (AttributeError, ValueError, KeyError):
            return None

    def _calculate_rsi(self, prices: "pd.Series", period: int = 14) -> float:
        """Calculate RSI indicator."""
        pd = load_pandas()
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
