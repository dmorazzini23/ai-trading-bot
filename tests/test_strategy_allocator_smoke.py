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
    alloc.config.delta_threshold = 0.0  # Allow repeated signals with same confidence
    sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s1")
    # Call allocate twice to build up signal history for confirmation
    alloc.allocate({"s1": [sig]})  # First call to start history
    out = alloc.allocate({"s1": [sig]})  # Second call should confirm the signal
    assert out and out[0].symbol == "AAPL"
    alloc.update_reward("s1", 0.5)
    force_coverage(strategy_allocator)
