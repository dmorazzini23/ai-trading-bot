"""Test for Alpaca import handling when packages are missing."""

import sys


def test_ai_trading_import_without_alpaca(monkeypatch):
    """Test that ai_trading can be imported even when alpaca packages are missing."""
    restore_modules: dict[str, object] = {}
    target_prefixes = ("alpaca", "ai_trading")
    for name, module in list(sys.modules.items()):
        if name == target_prefixes[0] or name.startswith(f"{target_prefixes[0]}."):
            restore_modules[name] = module
        elif name == target_prefixes[1] or name.startswith(f"{target_prefixes[1]}."):
            restore_modules[name] = module

    try:
        for name in list(restore_modules):
            sys.modules.pop(name, None)

        # Simulate missing Alpaca package.
        sys.modules["alpaca"] = None
        monkeypatch.setenv("TESTING", "true")

        # This should not raise an exception.
        import ai_trading
        import ai_trading.core.bot_engine

        # Check that ALPACA_AVAILABLE is False.
        assert hasattr(ai_trading.core.bot_engine, "ALPACA_AVAILABLE")
        assert ai_trading.core.bot_engine.ALPACA_AVAILABLE is False

        # Check that fallback classes are present.
        from ai_trading.core.bot_engine import OrderSide, TradingClient

        assert TradingClient is not None
        assert OrderSide is not None
    finally:
        for name in list(sys.modules):
            if name == target_prefixes[0] or name.startswith(f"{target_prefixes[0]}."):
                sys.modules.pop(name, None)
            elif name == target_prefixes[1] or name.startswith(f"{target_prefixes[1]}."):
                sys.modules.pop(name, None)
        sys.modules.update(restore_modules)


if __name__ == "__main__":
    test_ai_trading_import_without_alpaca()
