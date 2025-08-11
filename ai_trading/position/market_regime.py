"""
Market Regime Detector for adaptive position management.

Identifies current market conditions to adapt position holding strategies:
- Trending vs Range-bound markets
- High vs Low volatility regimes
- Bull vs Bear market phases
- Risk-on vs Risk-off sentiment

AI-AGENT-REF: Market regime detection for intelligent position management
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

# AI-AGENT-REF: graceful imports with fallbacks
# Use hard imports since numpy and pandas are dependencies
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""

    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


@dataclass
class RegimeMetrics:
    """Market regime analysis metrics."""

    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime classification
    trend_strength: float  # -1 to 1 (bear to bull)
    volatility_percentile: float  # 0-100 current vol vs historical
    momentum_score: float  # 0-1 momentum strength
    mean_reversion_score: float  # 0-1 mean reversion tendency
    regime_duration: int  # days in current regime
    timestamp: datetime


class MarketRegimeDetector:
    """
    Detect market regimes for adaptive position management.

    Uses multiple indicators to classify market conditions:
    - Price trend analysis (SMA slopes, ADX)
    - Volatility regime (VIX, ATR percentiles)
    - Momentum indicators (RSI, MACD)
    - Mean reversion metrics (Bollinger Band position)
    """

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.logger = logging.getLogger(__name__ + ".MarketRegimeDetector")

        # Regime detection parameters
        self.trend_lookback = 20  # days for trend analysis
        self.volatility_lookback = 60  # days for volatility percentile
        self.regime_history: list[RegimeMetrics] = []

        # Regime thresholds
        self.trending_threshold = 0.6  # ADX/trend strength threshold
        self.volatility_high_threshold = 75  # percentile threshold
        self.volatility_low_threshold = 25
        self.momentum_threshold = 0.7

    def detect_regime(self, symbol: str = "SPY") -> RegimeMetrics:
        """
        Detect current market regime for given symbol.

        Args:
            symbol: Market symbol to analyze (default SPY for broad market)

        Returns:
            RegimeMetrics with current regime classification
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if market_data is None or len(market_data) < self.volatility_lookback:
                return self._get_default_regime()

            # Calculate regime components
            trend_metrics = self._analyze_trend(market_data)
            volatility_metrics = self._analyze_volatility(market_data)
            momentum_metrics = self._analyze_momentum(market_data)
            mean_reversion_metrics = self._analyze_mean_reversion(market_data)

            # Classify regime
            regime = self._classify_regime(
                trend_metrics,
                volatility_metrics,
                momentum_metrics,
                mean_reversion_metrics,
            )

            # Calculate confidence
            confidence = self._calculate_confidence(
                trend_metrics, volatility_metrics, momentum_metrics
            )

            # Create regime metrics
            regime_metrics = RegimeMetrics(
                regime=regime,
                confidence=confidence,
                trend_strength=trend_metrics.get("strength", 0.0),
                volatility_percentile=volatility_metrics.get("percentile", 50.0),
                momentum_score=momentum_metrics.get("score", 0.5),
                mean_reversion_score=mean_reversion_metrics.get("score", 0.5),
                regime_duration=self._calculate_regime_duration(regime),
                timestamp=datetime.now(UTC),
            )

            # Store in history
            self.regime_history.append(regime_metrics)
            if len(self.regime_history) > 100:  # Keep last 100 regime readings
                self.regime_history.pop(0)

            self.logger.info(
                "REGIME_DETECTED | regime=%s confidence=%.2f trend=%.2f vol_pct=%.1f",
                regime.value,
                confidence,
                trend_metrics.get("strength", 0),
                volatility_metrics.get("percentile", 50),
            )

            return regime_metrics

        except Exception as exc:
            self.logger.warning("detect_regime failed: %s", exc)
            return self._get_default_regime()

    def get_regime_parameters(self, regime: MarketRegime) -> dict[str, float]:
        """
        Get position management parameters for given regime.

        Returns regime-specific parameters for:
        - Stop loss distances
        - Profit taking thresholds
        - Hold period adjustments
        - Position sizing modifiers
        """
        regime_params = {
            MarketRegime.TRENDING_BULL: {
                "stop_distance_multiplier": 1.5,  # Wider stops in trends
                "profit_taking_patience": 2.0,  # Let winners run longer
                "min_hold_days_multiplier": 1.5,
                "position_size_multiplier": 1.1,
                "trail_aggressiveness": 0.7,  # Less aggressive trailing
            },
            MarketRegime.TRENDING_BEAR: {
                "stop_distance_multiplier": 1.2,
                "profit_taking_patience": 0.8,  # Take profits quicker
                "min_hold_days_multiplier": 0.8,
                "position_size_multiplier": 0.9,
                "trail_aggressiveness": 0.9,  # More aggressive trailing
            },
            MarketRegime.RANGE_BOUND: {
                "stop_distance_multiplier": 0.8,  # Tighter stops in ranges
                "profit_taking_patience": 0.6,  # Quick profit taking
                "min_hold_days_multiplier": 0.7,
                "position_size_multiplier": 0.95,
                "trail_aggressiveness": 0.8,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "stop_distance_multiplier": 1.8,  # Much wider stops
                "profit_taking_patience": 0.5,  # Very quick profits
                "min_hold_days_multiplier": 0.5,
                "position_size_multiplier": 0.7,  # Smaller positions
                "trail_aggressiveness": 0.9,
            },
            MarketRegime.LOW_VOLATILITY: {
                "stop_distance_multiplier": 0.9,
                "profit_taking_patience": 1.5,  # Can be more patient
                "min_hold_days_multiplier": 1.2,
                "position_size_multiplier": 1.05,
                "trail_aggressiveness": 0.6,
            },
            MarketRegime.BREAKOUT: {
                "stop_distance_multiplier": 1.3,
                "profit_taking_patience": 1.8,  # Let breakouts run
                "min_hold_days_multiplier": 1.3,
                "position_size_multiplier": 1.0,
                "trail_aggressiveness": 0.5,  # Give room to run
            },
            MarketRegime.BREAKDOWN: {
                "stop_distance_multiplier": 0.9,
                "profit_taking_patience": 0.7,
                "min_hold_days_multiplier": 0.6,
                "position_size_multiplier": 0.8,
                "trail_aggressiveness": 0.95,  # Cut losses quickly
            },
        }

        return regime_params.get(regime, regime_params[MarketRegime.RANGE_BOUND])

    def _get_market_data(self, symbol: str) -> pd.DataFrame | None:
        """Get market data for regime analysis."""
        try:
            if self.ctx and hasattr(self.ctx, "data_fetcher"):
                # Try to get daily data for regime analysis
                df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
                if df is not None and not df.empty:
                    return df

            # Fallback: return None to trigger default regime
            return None

        except Exception:
            return None

    def _analyze_trend(self, data: pd.DataFrame) -> dict[str, float]:
        """Analyze price trend strength and direction."""
        try:
            if "close" not in data.columns or len(data) < self.trend_lookback:
                return {"strength": 0.0, "direction": 0.0}

            closes = data["close"].tail(self.trend_lookback)

            # Calculate trend using linear regression slope
            x = list(range(len(closes)))
            y = closes.tolist()

            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

            # Normalize slope to -1 to 1 range
            price_range = max(closes) - min(closes)
            if price_range > 0:
                normalized_slope = slope / price_range * len(closes)
                normalized_slope = max(-1.0, min(1.0, normalized_slope))
            else:
                normalized_slope = 0.0

            # Calculate trend strength (R-squared approximation)
            mean_y = sum_y / n
            ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
            y_pred = [slope * x[i] + (sum_y - slope * sum_x) / n for i in range(n)]
            ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))

            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trend_strength = max(0.0, min(1.0, r_squared))

            return {"strength": trend_strength, "direction": normalized_slope}

        except Exception:
            return {"strength": 0.0, "direction": 0.0}

    def _analyze_volatility(self, data: pd.DataFrame) -> dict[str, float]:
        """Analyze volatility regime using percentile ranking."""
        try:
            if "close" not in data.columns or len(data) < self.volatility_lookback:
                return {"percentile": 50.0, "current_vol": 0.0}

            closes = data["close"]

            # Calculate daily returns
            returns = closes.pct_change().dropna()

            if len(returns) < 20:
                return {"percentile": 50.0, "current_vol": 0.0}

            # Calculate rolling volatility (20-day)
            rolling_vol = returns.rolling(window=20).std() * (252**0.5)  # Annualized

            if len(rolling_vol) < self.volatility_lookback:
                return {"percentile": 50.0, "current_vol": 0.0}

            # Current volatility
            current_vol = (
                rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else 0.0
            )

            # Volatility percentile over lookback period
            vol_history = rolling_vol.tail(self.volatility_lookback).dropna()
            if len(vol_history) > 0:
                percentile = (vol_history < current_vol).mean() * 100
            else:
                percentile = 50.0

            return {"percentile": percentile, "current_vol": current_vol}

        except Exception:
            return {"percentile": 50.0, "current_vol": 0.0}

    def _analyze_momentum(self, data: pd.DataFrame) -> dict[str, float]:
        """Analyze momentum indicators for regime classification."""
        try:
            if "close" not in data.columns or len(data) < 30:
                return {"score": 0.5, "rsi": 50.0}

            closes = data["close"]

            # Calculate RSI (14-period)
            rsi = self._calculate_rsi(closes, 14)

            # Calculate momentum score (0-1 range)
            # RSI normalized to 0-1, with 50 as neutral
            rsi_score = rsi / 100.0 if not pd.isna(rsi) else 0.5

            # Price momentum (10-day vs 20-day)
            if len(closes) >= 20:
                recent_avg = closes.tail(10).mean()
                older_avg = closes.tail(20).head(10).mean()
                price_momentum = (
                    (recent_avg / older_avg - 1.0) if older_avg > 0 else 0.0
                )
                price_momentum = max(-0.5, min(0.5, price_momentum))  # Cap at +/-50%
                price_score = price_momentum + 0.5  # Normalize to 0-1
            else:
                price_score = 0.5

            # Combined momentum score
            momentum_score = (rsi_score * 0.6) + (price_score * 0.4)

            return {"score": momentum_score, "rsi": rsi if not pd.isna(rsi) else 50.0}

        except Exception:
            return {"score": 0.5, "rsi": 50.0}

    def _analyze_mean_reversion(self, data: pd.DataFrame) -> dict[str, float]:
        """Analyze mean reversion tendency."""
        try:
            if "close" not in data.columns or len(data) < 30:
                return {"score": 0.5}

            closes = data["close"]

            # Bollinger Band position (20-period, 2 std)
            sma = closes.rolling(window=20).mean()
            std = closes.rolling(window=20).std()

            current_price = closes.iloc[-1]
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]

            if pd.isna(current_sma) or pd.isna(current_std) or current_std == 0:
                return {"score": 0.5}

            # Position within Bollinger Bands (-1 to 1)
            bb_position = (current_price - current_sma) / (2 * current_std)
            bb_position = max(-1.0, min(1.0, bb_position))

            # Mean reversion score (higher when price is extreme)
            mean_reversion_score = abs(bb_position)

            return {"score": mean_reversion_score}

        except Exception:
            return {"score": 0.5}

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

        except Exception:
            return 50.0

    def _classify_regime(
        self,
        trend_metrics: dict,
        volatility_metrics: dict,
        momentum_metrics: dict,
        mean_reversion_metrics: dict,
    ) -> MarketRegime:
        """Classify market regime based on component analysis."""

        trend_strength = trend_metrics.get("strength", 0.0)
        trend_direction = trend_metrics.get("direction", 0.0)
        vol_percentile = volatility_metrics.get("percentile", 50.0)
        momentum_score = momentum_metrics.get("score", 0.5)

        # High/Low volatility regimes
        if vol_percentile > self.volatility_high_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif vol_percentile < self.volatility_low_threshold:
            return MarketRegime.LOW_VOLATILITY

        # Breakout/Breakdown detection
        if momentum_score > 0.8 and trend_strength > 0.6:
            return (
                MarketRegime.BREAKOUT if trend_direction > 0 else MarketRegime.BREAKDOWN
            )

        # Trending markets
        if trend_strength > self.trending_threshold:
            if trend_direction > 0.3:
                return MarketRegime.TRENDING_BULL
            elif trend_direction < -0.3:
                return MarketRegime.TRENDING_BEAR

        # Default to range-bound
        return MarketRegime.RANGE_BOUND

    def _calculate_confidence(
        self, trend_metrics: dict, volatility_metrics: dict, momentum_metrics: dict
    ) -> float:
        """Calculate confidence in regime classification."""
        try:
            trend_strength = trend_metrics.get("strength", 0.0)
            vol_percentile = volatility_metrics.get("percentile", 50.0)
            momentum_score = momentum_metrics.get("score", 0.5)

            # Base confidence on trend strength
            confidence = trend_strength

            # Boost confidence for extreme volatility
            vol_extremeness = abs(vol_percentile - 50) / 50.0
            confidence += vol_extremeness * 0.3

            # Boost confidence for strong momentum
            momentum_extremeness = abs(momentum_score - 0.5) * 2
            confidence += momentum_extremeness * 0.2

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5

    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long we've been in current regime."""
        try:
            if not self.regime_history:
                return 0

            duration = 0
            for metrics in reversed(self.regime_history):
                if metrics.regime == current_regime:
                    duration += 1
                else:
                    break

            return duration

        except Exception:
            return 0

    def _get_default_regime(self) -> RegimeMetrics:
        """Return default regime when detection fails."""
        return RegimeMetrics(
            regime=MarketRegime.RANGE_BOUND,
            confidence=0.3,
            trend_strength=0.0,
            volatility_percentile=50.0,
            momentum_score=0.5,
            mean_reversion_score=0.5,
            regime_duration=0,
            timestamp=datetime.now(UTC),
        )
