"""Tests for narrowed exceptions in RegimeDetector."""

from __future__ import annotations

from ai_trading.strategies.regime_detector import RegimeDetector
from ai_trading.risk.adaptive_sizing import MarketRegime


def test_detect_current_regime_handles_type_error() -> None:
    """Ensure TypeError is caught and fallback regime is returned."""  # AI-AGENT-REF: narrow exception test
    rd = RegimeDetector()
    market_data = {"prices": {"SPY": [1] * 70}, "returns": {"SPY": None}}
    regime, metrics = rd.detect_current_regime(market_data, index_symbol="SPY")
    assert isinstance(regime, MarketRegime)
    assert hasattr(metrics, "trend_strength")
