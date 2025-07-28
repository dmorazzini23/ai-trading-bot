import importlib
from strategies import TradeSignal
import sys

# Ensure clean import of strategy_allocator module
for module_name in list(sys.modules.keys()):
    if "strategy_allocator" in module_name:
        sys.modules.pop(module_name, None)

strategy_allocator = importlib.import_module("strategy_allocator")


def test_exit_confirmation():
    alloc = strategy_allocator.StrategyAllocator()
    buy = TradeSignal(symbol="A", side="buy", confidence=1.0, strategy="s")
    sell = TradeSignal(symbol="A", side="sell", confidence=1.0, strategy="s")
    out1 = alloc.allocate({"s": [buy]})
    assert any(s.side == "buy" for s in out1)
    out2 = alloc.allocate({"s": [sell]})
    assert not any(s.side == "sell" for s in out2)
    out3 = alloc.allocate({"s": [sell]})
    assert any(s.side == "sell" for s in out3)
