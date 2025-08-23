"""Tests for narrowed exceptions in risk metrics."""

from __future__ import annotations

from ai_trading.risk.metrics import RiskMetricsCalculator


def test_var_handles_bad_values_gracefully() -> None:
    """Invalid returns should return 0.0 instead of raising."""  # AI-AGENT-REF: narrow exception test
    rmc = RiskMetricsCalculator()
    returns = ["bad", None, {}]  # type: ignore
    assert rmc.calculate_var(returns, confidence_level=0.95) == 0.0
