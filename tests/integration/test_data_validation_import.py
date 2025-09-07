"""Integration test for data_validation package."""

from ai_trading.data_validation import MarketDataValidator, ValidationSeverity


def test_importable() -> None:
    """Package imports and exposes expected symbols."""
    validator = MarketDataValidator()
    assert isinstance(validator, MarketDataValidator)
    assert ValidationSeverity.ERROR.value == "error"
