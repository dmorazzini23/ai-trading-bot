"""
Market Regime Detection for Dynamic Trading Thresholds.

This module identifies market regimes (trending, ranging, volatile, crisis) to
dynamically adjust trading frequency and portfolio rebalancing thresholds.
"""

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import enhanced features if available
import numpy as np

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger
from ai_trading.risk.adaptive_sizing import MarketRegime, VolatilityRegime

NUMPY_AVAILABLE = True
ENHANCED_REGIMES_AVAILABLE = True


class TrendDirection(Enum):
    """Trend direction classification."""

    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class RegimeMetrics:
    """Market regime metrics and indicators."""

    trend_strength: float  # Trend strength (0-1)
    trend_direction: TrendDirection
    volatility_level: float  # Volatility percentile (0-1)
    volatility_regime: VolatilityRegime
    momentum: float  # Price momentum
    correlation_environment: float  # Average correlation level
    vix_level: float | None  # VIX level if available
    regime_confidence: float  # Confidence in regime classification

    def __post_init__(self):
        """Validate metrics are in expected ranges."""
        self.trend_strength = max(0.0, min(1.0, self.trend_strength))
        self.volatility_level = max(0.0, min(1.0, self.volatility_level))
        self.regime_confidence = max(0.0, min(1.0, self.regime_confidence))


@dataclass
class TradingThresholds:
    """Dynamic trading thresholds based on market regime."""

    rebalance_drift_threshold: float  # Portfolio drift threshold
    trade_frequency_multiplier: float  # Frequency adjustment multiplier
    correlation_penalty_adjustment: float  # Correlation penalty adjustment
    minimum_improvement_threshold: float  # Minimum improvement required
    safety_margin_multiplier: float  # Transaction cost safety margin

    def __post_init__(self):
        """Validate thresholds are positive."""
        self.rebalance_drift_threshold = max(
            0.01, min(0.5, self.rebalance_drift_threshold)
        )
        self.trade_frequency_multiplier = max(
            0.1, min(5.0, self.trade_frequency_multiplier)
        )
        self.minimum_improvement_threshold = max(
            0.001, min(0.1, self.minimum_improvement_threshold)
        )


class RegimeDetector:
    """
    Market regime detector for dynamic threshold adjustment.

    Analyzes market conditions to classify current regime and adjust
    trading parameters accordingly. Reduces frequency in choppy markets,
    allows higher frequency in trending markets.
    """

    def __init__(
        self,
        lookback_periods: int = 63,  # ~3 months of daily data
        trend_threshold: float = 0.02,  # 2% trend threshold
        volatility_window: int = 21,
    ):  # 1 month volatility window
        """
        Initialize regime detector.

        Args:
            lookback_periods: Number of periods for regime analysis
            trend_threshold: Minimum trend strength for trending regime
            volatility_window: Window for volatility calculation
        """
        self.lookback_periods = lookback_periods
        self.trend_threshold = trend_threshold
        self.volatility_window = volatility_window

        # Regime history for stability
        self.regime_history: list[MarketRegime] = []
        self.max_history_length = 10

        # Volatility percentile thresholds
        self.volatility_thresholds = {
            "extremely_low": 0.10,
            "low": 0.25,
            "normal_low": 0.40,
            "normal_high": 0.60,
            "high": 0.75,
            "extremely_high": 0.90,
        }

        logger.info(
            f"RegimeDetector initialized with lookback={lookback_periods}, "
            f"trend_threshold={trend_threshold:.1%}"
        )

    def detect_current_regime(
        self, market_data: dict[str, Any], index_symbol: str = "SPY"
    ) -> tuple[MarketRegime, RegimeMetrics]:
        """
        Detect current market regime based on market data.

        Args:
            market_data: Market data including prices, volumes, volatility
            index_symbol: Index symbol to use for regime detection (default SPY)

        Returns:
            Tuple of (detected_regime, detailed_metrics)
        """
        try:
            # Extract price data for analysis
            prices = market_data.get("prices", {})
            returns_data = market_data.get("returns", {})
            market_data.get("volumes", {})

            # Use index symbol or fallback to any available symbol
            analysis_symbol = (
                index_symbol
                if index_symbol in prices
                else next(iter(prices.keys()), None)
            )

            if not analysis_symbol or analysis_symbol not in returns_data:
                logger.warning("Insufficient data for regime detection, using fallback")
                return self._fallback_regime_detection()

            # Calculate regime metrics
            returns = returns_data[analysis_symbol][-self.lookback_periods :]

            if len(returns) < self.volatility_window:
                logger.warning("Insufficient return data for regime detection")
                return self._fallback_regime_detection()

            # Calculate trend metrics
            trend_strength, trend_direction = self._calculate_trend_metrics(returns)

            # Calculate volatility metrics
            volatility_level, volatility_regime = self._calculate_volatility_regime(
                returns
            )

            # Calculate momentum
            momentum = self._calculate_momentum(returns)

            # Calculate correlation environment
            correlation_environment = self._calculate_correlation_environment(
                market_data
            )

            # Get VIX level if available
            vix_level = self._get_vix_level(market_data)

            # Calculate regime confidence
            regime_confidence = self._calculate_regime_confidence(
                returns, trend_strength, volatility_level
            )

            # Create metrics object
            metrics = RegimeMetrics(
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                volatility_level=volatility_level,
                volatility_regime=volatility_regime,
                momentum=momentum,
                correlation_environment=correlation_environment,
                vix_level=vix_level,
                regime_confidence=regime_confidence,
            )

            # Determine overall regime
            regime = self._classify_market_regime(metrics)

            # Update regime history for stability
            self._update_regime_history(regime)

            logger.info(
                f"Market regime detected: {regime.value} "
                f"(trend={trend_strength:.3f}, vol={volatility_level:.3f}, "
                f"confidence={regime_confidence:.3f})"
            )

            return regime, metrics

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Error detecting market regime: {e}")
            return self._fallback_regime_detection()

    def calculate_dynamic_thresholds(
        self,
        regime: MarketRegime,
        metrics: RegimeMetrics,
        base_thresholds: dict[str, float] | None = None,
    ) -> TradingThresholds:
        """
        Calculate dynamic trading thresholds based on market regime.

        Args:
            regime: Detected market regime
            metrics: Detailed regime metrics
            base_thresholds: Base threshold values to adjust

        Returns:
            Adjusted trading thresholds
        """
        try:
            # Default base thresholds
            if base_thresholds is None:
                base_thresholds = {
                    "rebalance_drift_threshold": 0.05,
                    "trade_frequency_multiplier": 1.0,
                    "correlation_penalty_adjustment": 1.0,
                    "minimum_improvement_threshold": 0.02,
                    "safety_margin_multiplier": 2.0,
                }

            # Regime-specific adjustments
            adjustments = self._get_regime_adjustments(regime, metrics)

            # Apply adjustments to base thresholds
            adjusted_thresholds = TradingThresholds(
                rebalance_drift_threshold=base_thresholds["rebalance_drift_threshold"]
                * adjustments["drift_multiplier"],
                trade_frequency_multiplier=base_thresholds["trade_frequency_multiplier"]
                * adjustments["frequency_multiplier"],
                correlation_penalty_adjustment=base_thresholds[
                    "correlation_penalty_adjustment"
                ]
                * adjustments["correlation_multiplier"],
                minimum_improvement_threshold=base_thresholds[
                    "minimum_improvement_threshold"
                ]
                * adjustments["improvement_multiplier"],
                safety_margin_multiplier=base_thresholds["safety_margin_multiplier"]
                * adjustments["safety_multiplier"],
            )

            logger.info(
                f"Dynamic thresholds for {regime.value}: "
                f"drift={adjusted_thresholds.rebalance_drift_threshold:.3f}, "
                f"freq={adjusted_thresholds.trade_frequency_multiplier:.2f}"
            )

            return adjusted_thresholds

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Error calculating dynamic thresholds: {e}")
            # Return conservative defaults
            return TradingThresholds(0.03, 0.5, 1.5, 0.03, 3.0)

    def _calculate_trend_metrics(
        self, returns: list[float]
    ) -> tuple[float, TrendDirection]:
        """Calculate trend strength and direction from returns."""
        try:
            if len(returns) < 10:
                return 0.0, TrendDirection.SIDEWAYS

            # Calculate cumulative returns
            sum(returns)

            # Calculate trend strength using linear regression slope
            if NUMPY_AVAILABLE:
                x = np.arange(len(returns))
                y = np.cumsum(returns)
                slope = np.polyfit(x, y, 1)[0]
            else:
                # Fallback: simple slope calculation
                n = len(returns)
                x_mean = (n - 1) / 2
                y_values = []
                cum_return = 0
                for r in returns:
                    cum_return += r
                    y_values.append(cum_return)
                y_mean = sum(y_values) / n

                numerator = sum((i - x_mean) * (y_values[i] - y_mean) for i in range(n))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                slope = numerator / denominator if denominator > 0 else 0

            # Normalize trend strength
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.01
            trend_strength = min(
                1.0, abs(slope) / max(volatility * math.sqrt(len(returns)), 0.001)
            )

            # Classify trend direction
            if slope > self.trend_threshold:
                if slope > self.trend_threshold * 2:
                    direction = TrendDirection.STRONG_UPTREND
                else:
                    direction = TrendDirection.UPTREND
            elif slope < -self.trend_threshold:
                if slope < -self.trend_threshold * 2:
                    direction = TrendDirection.STRONG_DOWNTREND
                else:
                    direction = TrendDirection.DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS

            return trend_strength, direction

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Error calculating trend metrics: {e}")
            return 0.0, TrendDirection.SIDEWAYS

    def _calculate_volatility_regime(
        self, returns: list[float]
    ) -> tuple[float, VolatilityRegime]:
        """Calculate volatility level and regime classification."""
        try:
            if len(returns) < self.volatility_window:
                return 0.5, VolatilityRegime.NORMAL

            # Calculate rolling volatilities for percentile ranking
            rolling_vols = []
            for i in range(self.volatility_window, len(returns) + 1):
                window_returns = returns[i - self.volatility_window : i]
                if len(window_returns) > 1:
                    vol = statistics.stdev(window_returns) * math.sqrt(
                        252
                    )  # Annualized
                    rolling_vols.append(vol)

            if not rolling_vols:
                return 0.5, VolatilityRegime.NORMAL

            # Current volatility
            current_vol = rolling_vols[-1]

            # Calculate percentile rank
            sorted_vols = sorted(rolling_vols)
            rank = len([v for v in sorted_vols if v <= current_vol])
            percentile = rank / len(sorted_vols)

            # Classify volatility regime
            if percentile <= self.volatility_thresholds["extremely_low"]:
                regime = VolatilityRegime.EXTREMELY_LOW
            elif percentile <= self.volatility_thresholds["low"]:
                regime = VolatilityRegime.LOW
            elif percentile <= self.volatility_thresholds["high"]:
                regime = VolatilityRegime.NORMAL
            elif percentile <= self.volatility_thresholds["extremely_high"]:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.EXTREMELY_HIGH

            return percentile, regime

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Error calculating volatility regime: {e}")
            return 0.5, VolatilityRegime.NORMAL

    def _calculate_momentum(self, returns: list[float]) -> float:
        """Calculate price momentum indicator."""
        try:
            if len(returns) < 21:  # Need at least 1 month
                return 0.0

            # Short-term momentum (last 5 days)
            short_momentum = sum(returns[-5:]) if len(returns) >= 5 else 0

            # Medium-term momentum (last 21 days)
            medium_momentum = sum(returns[-21:]) if len(returns) >= 21 else 0

            # Weighted momentum score
            momentum = short_momentum * 0.3 + medium_momentum * 0.7

            # Normalize to [-1, 1] range
            momentum = max(
                -1.0, min(1.0, momentum / 0.1)
            )  # Assume 10% is extreme momentum

            return momentum

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Error calculating momentum: {e}")
            return 0.0

    def _calculate_correlation_environment(self, market_data: dict[str, Any]) -> float:
        """Calculate average correlation environment."""
        try:
            correlation_matrix = market_data.get("correlations", {})

            if not correlation_matrix:
                return 0.3  # Default moderate correlation

            # Calculate average absolute correlation
            correlations = []
            for symbol1, corr_dict in correlation_matrix.items():
                for symbol2, correlation in corr_dict.items():
                    if symbol1 != symbol2:  # Exclude self-correlation
                        correlations.append(abs(correlation))

            if not correlations:
                return 0.3

            avg_correlation = statistics.mean(correlations)
            return min(1.0, max(0.0, avg_correlation))

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Error calculating correlation environment: {e}")
            return 0.3

    def _get_vix_level(self, market_data: dict[str, Any]) -> float | None:
        """Get VIX level if available."""
        try:
            prices = market_data.get("prices", {})
            vix_symbols = ["VIX", "^VIX", "VIXY"]

            for symbol in vix_symbols:
                if symbol in prices:
                    return prices[symbol]

            return None

        except (ValueError, TypeError, KeyError):  # AI-AGENT-REF: narrow exception
            return None

    def _calculate_regime_confidence(
        self, returns: list[float], trend_strength: float, volatility_level: float
    ) -> float:
        """Calculate confidence in regime classification."""
        try:
            confidence = 1.0

            # Penalize short data series
            if len(returns) < self.lookback_periods:
                confidence *= len(returns) / self.lookback_periods

            # Higher confidence for strong trends or extreme volatility
            if trend_strength > 0.7 or volatility_level > 0.8 or volatility_level < 0.2:
                confidence *= 1.2

            # Lower confidence for moderate/unclear conditions
            if 0.3 < trend_strength < 0.7 and 0.3 < volatility_level < 0.7:
                confidence *= 0.8

            return max(0.0, min(1.0, confidence))

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ):  # AI-AGENT-REF: narrow exception
            return 0.5

    def _classify_market_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """Classify overall market regime from metrics."""
        try:
            # Crisis detection (high volatility + negative momentum)
            if (
                metrics.volatility_regime
                in [VolatilityRegime.EXTREMELY_HIGH, VolatilityRegime.HIGH]
                and metrics.momentum < -0.5
            ):
                return MarketRegime.CRISIS

            # High volatility regime
            if metrics.volatility_regime == VolatilityRegime.EXTREMELY_HIGH:
                return MarketRegime.HIGH_VOLATILITY

            # Low volatility regime
            if metrics.volatility_regime == VolatilityRegime.EXTREMELY_LOW:
                return MarketRegime.LOW_VOLATILITY

            # Trending regimes
            if metrics.trend_strength > 0.6:
                if metrics.trend_direction in [
                    TrendDirection.STRONG_UPTREND,
                    TrendDirection.UPTREND,
                ]:
                    return MarketRegime.BULL_TRENDING
                elif metrics.trend_direction in [
                    TrendDirection.STRONG_DOWNTREND,
                    TrendDirection.DOWNTREND,
                ]:
                    return MarketRegime.BEAR_TRENDING

            # Sideways/ranging market
            if (
                metrics.trend_strength < 0.3
                and metrics.volatility_regime == VolatilityRegime.NORMAL
            ):
                return MarketRegime.SIDEWAYS_RANGE

            # Default to normal
            return MarketRegime.NORMAL

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            statistics.StatisticsError,
            np.linalg.LinAlgError,
        ) as e:  # AI-AGENT-REF: narrow exception
            logger.error(f"Error classifying market regime: {e}")
            return MarketRegime.NORMAL

    def _get_regime_adjustments(
        self, regime: MarketRegime, metrics: RegimeMetrics
    ) -> dict[str, float]:
        """Get threshold adjustment multipliers for each regime."""
        # Base adjustments by regime
        regime_adjustments = {
            MarketRegime.BULL_TRENDING: {
                "drift_multiplier": 1.2,  # Allow more drift in trends
                "frequency_multiplier": 1.3,  # Allow more trading
                "correlation_multiplier": 0.8,  # Reduce correlation penalty
                "improvement_multiplier": 0.8,  # Lower improvement threshold
                "safety_multiplier": 1.0,  # Normal safety margin
            },
            MarketRegime.BEAR_TRENDING: {
                "drift_multiplier": 1.1,
                "frequency_multiplier": 1.1,
                "correlation_multiplier": 0.9,
                "improvement_multiplier": 1.0,
                "safety_multiplier": 1.2,  # Higher safety in bear markets
            },
            MarketRegime.SIDEWAYS_RANGE: {
                "drift_multiplier": 0.8,  # Tighter drift control
                "frequency_multiplier": 0.7,  # Reduce trading frequency
                "correlation_multiplier": 1.2,  # Higher correlation penalty
                "improvement_multiplier": 1.2,  # Higher improvement threshold
                "safety_multiplier": 1.5,  # Higher safety margin
            },
            MarketRegime.HIGH_VOLATILITY: {
                "drift_multiplier": 0.6,  # Very tight control
                "frequency_multiplier": 0.5,  # Much lower frequency
                "correlation_multiplier": 1.5,  # Much higher correlation penalty
                "improvement_multiplier": 1.5,  # Much higher threshold
                "safety_multiplier": 2.0,  # Much higher safety
            },
            MarketRegime.LOW_VOLATILITY: {
                "drift_multiplier": 1.3,  # Allow more drift
                "frequency_multiplier": 1.2,  # Allow more trading
                "correlation_multiplier": 0.9,  # Slight reduction
                "improvement_multiplier": 0.9,  # Slight reduction
                "safety_multiplier": 0.8,  # Lower safety margin
            },
            MarketRegime.CRISIS: {
                "drift_multiplier": 0.3,  # Extremely tight control
                "frequency_multiplier": 0.2,  # Minimal trading
                "correlation_multiplier": 2.0,  # Maximum correlation penalty
                "improvement_multiplier": 2.0,  # Very high threshold
                "safety_multiplier": 3.0,  # Maximum safety
            },
            MarketRegime.NORMAL: {
                "drift_multiplier": 1.0,  # No adjustment
                "frequency_multiplier": 1.0,
                "correlation_multiplier": 1.0,
                "improvement_multiplier": 1.0,
                "safety_multiplier": 1.0,
            },
        }

        return regime_adjustments.get(regime, regime_adjustments[MarketRegime.NORMAL])

    def _update_regime_history(self, regime: MarketRegime):
        """Update regime history for stability analysis."""
        self.regime_history.append(regime)
        if len(self.regime_history) > self.max_history_length:
            self.regime_history.pop(0)

    def _fallback_regime_detection(self) -> tuple[MarketRegime, RegimeMetrics]:
        """Fallback regime detection when data is insufficient."""
        logger.warning("Using fallback regime detection")

        metrics = RegimeMetrics(
            trend_strength=0.0,
            trend_direction=TrendDirection.SIDEWAYS,
            volatility_level=0.5,
            volatility_regime=VolatilityRegime.NORMAL,
            momentum=0.0,
            correlation_environment=0.3,
            vix_level=None,
            regime_confidence=0.3,
        )

        return MarketRegime.NORMAL, metrics


# AI-AGENT-REF: Market regime detection for dynamic portfolio optimization
def create_regime_detector(config: dict[str, Any] | None = None) -> RegimeDetector:
    """
    Factory function to create regime detector with configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured RegimeDetector instance
    """
    if config is None:
        config = {}

    return RegimeDetector(
        lookback_periods=config.get("lookback_periods", 63),
        trend_threshold=config.get("trend_threshold", 0.02),
        volatility_window=config.get("volatility_window", 21),
    )
