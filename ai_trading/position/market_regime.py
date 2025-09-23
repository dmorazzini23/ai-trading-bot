from __future__ import annotations

"""Lightweight market regime utilities for tests.

This module intentionally provides a minimal implementation that supports the
unit tests without pulling in heavy dependencies or complex logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai_trading.logging import get_logger


class MarketRegime(Enum):
    """Basic market regime classifications."""

    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Lightweight metrics summary for detected market regimes."""

    regime: MarketRegime = MarketRegime.RANGE_BOUND
    confidence: float = 0.0
    trend_strength: float = 0.0
    volatility_percentile: float = 0.0


class MarketRegimeDetector:
    """Very small regime detector used in tests."""

    _DEFAULT_CONFIDENCE = 0.2

    def __init__(self, ctx: Any | None = None) -> None:
        self.ctx = ctx
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        """Clamp a float to the provided bounds."""

        return max(lower, min(upper, value))

    def _classify_regime(
        self,
        trend_metrics: dict[str, float],
        vol_metrics: dict[str, float],
        momentum_metrics: dict[str, float],
        mean_reversion_metrics: dict[str, float],
    ) -> MarketRegime:
        """Classify a market regime using simple heuristics."""

        vol_pct = vol_metrics.get("percentile", 0.0)
        if vol_pct >= 80.0:
            return MarketRegime.HIGH_VOLATILITY

        strength = trend_metrics.get("strength", 0.0)
        direction = trend_metrics.get("direction", 0.0)
        if strength >= 0.6:
            return MarketRegime.TRENDING_BULL if direction >= 0 else MarketRegime.TRENDING_BEAR
        return MarketRegime.RANGE_BOUND

    def detect_regime(
        self,
        trend_metrics: dict[str, float] | None = None,
        vol_metrics: dict[str, float] | None = None,
        momentum_metrics: dict[str, float] | None = None,
        mean_reversion_metrics: dict[str, float] | None = None,
    ) -> RegimeMetrics:
        """Produce a lightweight regime assessment for the position manager."""

        trend_metrics = trend_metrics or {}
        vol_metrics = vol_metrics or {}
        momentum_metrics = momentum_metrics or {}
        mean_reversion_metrics = mean_reversion_metrics or {}

        trend_strength = float(trend_metrics.get("strength", 0.0) or 0.0)
        volatility_percentile = float(vol_metrics.get("percentile", 0.0) or 0.0)

        has_signal = any((trend_metrics, vol_metrics, momentum_metrics, mean_reversion_metrics))
        if not has_signal:
            self.logger.debug("detect_regime called without metrics; returning defaults")
            return RegimeMetrics(
                regime=MarketRegime.RANGE_BOUND,
                confidence=self._DEFAULT_CONFIDENCE,
                trend_strength=0.0,
                volatility_percentile=0.0,
            )

        regime = self._classify_regime(
            trend_metrics, vol_metrics, momentum_metrics, mean_reversion_metrics
        )

        confidence_components: list[float] = []
        if "strength" in trend_metrics:
            confidence_components.append(self._clamp(abs(trend_strength)))
        if "percentile" in vol_metrics:
            confidence_components.append(self._clamp(volatility_percentile / 100.0))
        if "score" in momentum_metrics:
            momentum_score = float(momentum_metrics.get("score", 0.0) or 0.0)
            confidence_components.append(self._clamp(abs(momentum_score)))
        if "score" in mean_reversion_metrics:
            mean_rev_score = float(mean_reversion_metrics.get("score", 0.0) or 0.0)
            # Scores close to 0.5 imply balanced mean-reversion; favor extremes less.
            balance = 1.0 - min(abs(mean_rev_score - 0.5) * 2.0, 1.0)
            confidence_components.append(self._clamp(balance))

        if confidence_components:
            confidence = sum(confidence_components) / len(confidence_components)
        else:
            confidence = self._DEFAULT_CONFIDENCE

        if regime in (MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR):
            confidence = max(confidence, self._clamp(trend_strength))
        elif regime is MarketRegime.HIGH_VOLATILITY:
            high_vol_conf = (volatility_percentile - 50.0) / 50.0
            confidence = max(confidence, self._clamp(high_vol_conf))

        confidence = self._clamp(confidence)

        regime_metrics = RegimeMetrics(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_percentile=volatility_percentile,
        )

        self.logger.debug(
            "detect_regime classified %s with confidence %.2f", regime_metrics.regime, regime_metrics.confidence
        )
        return regime_metrics

    def get_regime_parameters(self, regime: MarketRegime) -> dict[str, float]:
        """Return heuristic parameters for a regime."""

        if regime is MarketRegime.TRENDING_BULL:
            return {
                "stop_distance_multiplier": 1.5,
                "profit_taking_patience": 2.0,
                "position_size_multiplier": 1.0,
                "trail_aggressiveness": 1.0,
            }
        if regime is MarketRegime.HIGH_VOLATILITY:
            return {
                "stop_distance_multiplier": 0.7,
                "profit_taking_patience": 0.8,
                "position_size_multiplier": 0.5,
                "trail_aggressiveness": 1.5,
            }
        return {
            "stop_distance_multiplier": 1.0,
            "profit_taking_patience": 1.0,
            "position_size_multiplier": 1.0,
            "trail_aggressiveness": 1.0,
        }


def detect_market_regime(*_args: Any, **_kwargs: Any) -> MarketRegime:
    """Placeholder entry point for compatibility."""

    return MarketRegime.UNKNOWN
