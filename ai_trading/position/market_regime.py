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
    """Placeholder metrics container."""

    trend_strength: float = 0.0
    volatility_percentile: float = 0.0


class MarketRegimeDetector:
    """Very small regime detector used in tests."""

    def __init__(self, ctx: Any | None = None) -> None:
        self.ctx = ctx
        self.logger = get_logger(self.__class__.__name__)

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
