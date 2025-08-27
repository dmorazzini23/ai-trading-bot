"""Test for Alpaca import handling when packages are missing."""

import sys


def test_ai_trading_import_without_alpaca():
    """Test that ai_trading can be imported even when alpaca packages are missing."""
    # Remove alpaca modules from sys.modules to simulate missing package
    alpaca_modules = [module for module in sys.modules.keys() if 'alpaca' in module.lower()]
    for module in alpaca_modules:
        sys.modules.pop(module, None)

    # Simulate missing Alpaca package by setting it to None
    sys.modules['alpaca'] = None

    # Set testing mode
    import os
    os.environ['TESTING'] = 'true'

    try:
        # This should not raise an exception
        import ai_trading
        import ai_trading.alpaca_api as api

        # Check that ALPACA_AVAILABLE is False
        assert hasattr(api, 'ALPACA_AVAILABLE')
        assert api.ALPACA_AVAILABLE is False
    finally:
        # Clean up environment
        os.environ.pop('TESTING', None)

        # Restore original modules (though this won't actually restore them)
        for module in alpaca_modules:
            sys.modules.pop(module, None)


if __name__ == "__main__":
    test_ai_trading_import_without_alpaca()
