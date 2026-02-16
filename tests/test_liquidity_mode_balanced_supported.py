from __future__ import annotations

from ai_trading.execution.liquidity import LiquidityAnalyzer


def test_liquidity_analyzer_supports_balanced_key() -> None:
    analyzer = LiquidityAnalyzer()
    assert "balanced" in analyzer.participation_thresholds
    assert "moderate" not in analyzer.participation_thresholds
