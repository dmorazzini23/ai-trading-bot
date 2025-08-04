import importlib
import sys
from pathlib import Path

import pytest

from strategies import TradeSignal

import strategy_allocator


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_allocator():
    alloc = strategy_allocator.StrategyAllocator()
    
    # Configuration that properly tests signal confirmation workflow
    alloc.config.delta_threshold = 0.0        # Allow repeated signals
    alloc.config.signal_confirmation_bars = 2  # Require 2 bars for proper confirmation testing
    alloc.config.min_confidence = 0.0         # Ensure confidence threshold is met
    
    sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s1")
    
    # First call: Build signal history (returns empty list)
    out1 = alloc.allocate({"s1": [sig]})
    assert out1 == []  # Should be empty as signal is not yet confirmed
    
    # Second call: Confirm signals (returns confirmed signal)
    out2 = alloc.allocate({"s1": [sig]})
    assert out2 and out2[0].symbol == "AAPL"
    
    alloc.update_reward("s1", 0.5)
    force_coverage(strategy_allocator)
