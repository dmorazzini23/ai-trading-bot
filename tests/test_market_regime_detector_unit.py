"""Unit tests for the lightweight MarketRegimeDetector utilities."""

import pytest

from ai_trading.position.market_regime import MarketRegime, MarketRegimeDetector, RegimeMetrics


def test_detect_regime_happy_path():
    """detect_regime should return a populated metrics container when data is provided."""

    detector = MarketRegimeDetector()
    metrics = detector.detect_regime(
        trend_metrics={'strength': 0.7, 'direction': 0.3},
        vol_metrics={'percentile': 40.0},
        momentum_metrics={'score': 0.6},
        mean_reversion_metrics={'score': 0.2},
    )

    assert isinstance(metrics, RegimeMetrics)
    assert metrics.regime == MarketRegime.TRENDING_BULL
    assert 0.6 <= metrics.confidence <= 1.0
    assert metrics.trend_strength == 0.7
    assert metrics.volatility_percentile == 40.0


def test_detect_regime_fallback_defaults():
    """detect_regime should return safe defaults when no metrics are supplied."""

    detector = MarketRegimeDetector()
    metrics = detector.detect_regime()

    assert isinstance(metrics, RegimeMetrics)
    assert metrics.regime == MarketRegime.RANGE_BOUND
    assert metrics.confidence == pytest.approx(0.2)
    assert metrics.trend_strength == 0.0
    assert metrics.volatility_percentile == 0.0
