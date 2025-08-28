import sys

import pytest
from ai_trading.strategies import TradeSignal


@pytest.fixture(autouse=True)
def ensure_real_strategy_allocator():
    """Ensure we get the real strategy_allocator module, not a mock."""
    # Save the original module if it exists
    original_module = sys.modules.get('strategy_allocator')

    # Remove any existing mock module
    if 'strategy_allocator' in sys.modules:
        del sys.modules['strategy_allocator']

    # Import the real module from file path
    from ai_trading import strategy_allocator  # AI-AGENT-REF: normalized import

    # Verify we have the real StrategyAllocator class
    assert hasattr(strategy_allocator.StrategyAllocator, '__init__')
    assert hasattr(strategy_allocator.StrategyAllocator(), 'config')

    yield strategy_allocator

    # Cleanup: restore original module if it was there
    if original_module and hasattr(original_module, 'StrategyAllocator'):
        if original_module.StrategyAllocator != object:
            sys.modules['strategy_allocator'] = original_module


def test_exit_confirmation(ensure_real_strategy_allocator):
    strategy_allocator = ensure_real_strategy_allocator
    alloc = strategy_allocator.StrategyAllocator()
    # Explicitly set configuration to ensure test isolation
    alloc.replace_config(
        delta_threshold=0.0,  # Allow repeated signals with same confidence
        signal_confirmation_bars=2,  # Ensure we have expected confirmation bars
    )

    buy = TradeSignal(symbol="A", side="buy", confidence=1.0, strategy="s")
    sell = TradeSignal(symbol="A", side="sell", confidence=1.0, strategy="s")

    # Need to call allocate twice to confirm signals (signal_confirmation_bars = 2)
    alloc.select_signals({"s": [buy]})  # First call to build history
    out1 = alloc.select_signals({"s": [buy]})  # Second call should confirm and set hold_protect=4
    assert any(s.side == "buy" for s in out1)

    # Now try to sell - should be blocked by hold protection 4 times
    alloc.select_signals({"s": [sell]})  # First sell call - builds history
    out2 = alloc.select_signals({"s": [sell]})  # Second sell call - confirmed but blocked by hold_protect (remaining=3)
    assert not any(s.side == "sell" for s in out2)

    # Need to call sell 3 more times to exhaust hold protection
    out3 = alloc.select_signals({"s": [sell]})  # Third sell call - blocked by hold_protect (remaining=2)
    assert not any(s.side == "sell" for s in out3)

    out4 = alloc.select_signals({"s": [sell]})  # Fourth sell call - blocked by hold_protect (remaining=1)
    assert not any(s.side == "sell" for s in out4)

    out5 = alloc.select_signals({"s": [sell]})  # Fifth sell call - blocked by hold_protect (remaining=0)
    assert not any(s.side == "sell" for s in out5)

    out6 = alloc.select_signals({"s": [sell]})  # Sixth sell call - should finally go through
    assert any(s.side == "sell" for s in out6)
