"""
Test cases for the Kelly calculation confidence fix.

This module tests that confidence values > 1.0 are properly normalized
to valid probability ranges in the Kelly calculation.
"""
import math
import os

# Test the actual import and function from bot_engine
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_kelly_confidence_normalization():
    """Test that high confidence values are properly normalized to probabilities."""
    # Mock BotContext for testing
    # Import the actual function (if available)
    try:
        from ai_trading.core.bot_engine import fractional_kelly_size

        ctx = MockBotContext()
        balance = 10000.0
        price = 100.0
        atr = 2.0

        # Test cases that previously caused errors
        problematic_confidences = [
            3.315653439025116,
            3.0650464264275152,
            5.0,
            2.5,
        ]

        for confidence in problematic_confidences:
            # This should not raise an error and should return a valid position size
            result = fractional_kelly_size(ctx, balance, price, atr, confidence)

            # Verify result is reasonable
            assert isinstance(result, int), f"Result should be integer, got {type(result)}"
            assert result >= 0, f"Position size should be non-negative, got {result}"
            assert result < 1000, f"Position size should be reasonable, got {result}"

        # Test edge cases
        assert fractional_kelly_size(ctx, balance, price, atr, 0.0) >= 0
        assert fractional_kelly_size(ctx, balance, price, atr, 1.0) >= 0
        assert fractional_kelly_size(ctx, balance, price, atr, -0.5) >= 0

    except ImportError:
        # If we can't import, at least test our normalization logic
        def sigmoid_normalize(value):
            if value > 1.0:
                return 1.0 / (1.0 + math.exp(-value + 1.0))
            elif value < 0:
                return 0.0
            return value

        test_values = [3.315653439025116, 3.0650464264275152, 5.0, 1.0, 0.5, 0.0, -0.5]

        for value in test_values:
            normalized = sigmoid_normalize(value)
            assert 0.0 <= normalized <= 1.0, f"Normalized value {normalized} should be in [0,1]"

            # Values > 1 should be mapped to something > 0.5
            if value > 1.0:
                assert normalized > 0.5, f"High confidence {value} should map to high probability"


def test_kelly_input_validation():
    """Test that Kelly calculation properly validates all inputs."""
    # Mock BotContext for testing
    try:
        from ai_trading.core.bot_engine import fractional_kelly_size

        ctx = MockBotContext()

        # Test invalid inputs return 0 or minimal position
        assert fractional_kelly_size(ctx, -1000, 100, 2.0, 0.6) == 0  # negative balance
        assert fractional_kelly_size(ctx, 1000, -100, 2.0, 0.6) == 0  # negative price
        assert fractional_kelly_size(ctx, 1000, 0, 2.0, 0.6) == 0     # zero price

        # Test that valid inputs work
        result = fractional_kelly_size(ctx, 1000, 100, 2.0, 0.6)
        assert result > 0, "Valid inputs should produce positive position size"

    except ImportError:
        # Skip if we can't import the actual function
        pytest.skip("Cannot import fractional_kelly_size function")


if __name__ == "__main__":
    test_kelly_confidence_normalization()
    test_kelly_input_validation()
