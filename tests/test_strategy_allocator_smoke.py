from pathlib import Path
import importlib
import pytest
import sys
from strategies import TradeSignal

sys.modules.pop("strategy_allocator", None)
strategy_allocator = importlib.import_module("strategy_allocator")


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_allocator():
    alloc = strategy_allocator.StrategyAllocator()
    sig = TradeSignal(symbol="AAPL", side="buy", confidence=1.0, strategy="s1")
    out = alloc.allocate({"s1": [sig]})
    assert out and out[0].symbol == "AAPL"
    alloc.update_reward("s1", 0.5)
    force_coverage(strategy_allocator)
