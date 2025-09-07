"""Tests for institutional constants."""

from ai_trading.institutional.constants import MAX_KELLY_FRACTION


def test_max_kelly_fraction():
    """Ensure institutional Kelly fraction meets spec."""
    assert MAX_KELLY_FRACTION == 0.25
