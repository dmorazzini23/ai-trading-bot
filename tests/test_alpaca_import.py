"""Test for Alpaca import handling when packages are missing."""

import sys


def test_ai_trading_import_without_alpaca():
    """Test that ai_trading can be imported even when alpaca packages are missing."""
    # Remove alpaca modules from sys.modules to simulate missing packages
    alpaca_modules = [module for module in sys.modules.keys() if 'alpaca' in module.lower()]
    for module in alpaca_modules:
        sys.modules.pop(module, None)

    # Simulate missing alpaca packages by setting them to None
    sys.modules['alpaca_trade_api'] = None
    sys.modules['alpaca.trading'] = None
    sys.modules['alpaca.data'] = None
    sys.modules['alpaca'] = None

    # Set testing mode
    import os
    os.environ['TESTING'] = 'true'

    try:
        # This should not raise an exception
        import ai_trading
        import ai_trading.core.bot_engine

        # Check that ALPACA_AVAILABLE is False
        assert hasattr(ai_trading.core.bot_engine, 'ALPACA_AVAILABLE')
        assert ai_trading.core.bot_engine.ALPACA_AVAILABLE is False

        # Check that mock classes are used
        from ai_trading.core.bot_engine import OrderSide, TradingClient
        assert TradingClient is not None
        assert OrderSide is not None

        print("SUCCESS: ai_trading imported successfully without Alpaca packages")

    finally:
        # Clean up environment
        os.environ.pop('TESTING', None)

        # Restore original modules (though this won't actually restore them)
        for module in alpaca_modules:
            sys.modules.pop(module, None)


if __name__ == "__main__":
    test_ai_trading_import_without_alpaca()
