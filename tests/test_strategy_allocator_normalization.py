from types import SimpleNamespace

from ai_trading.strategy_allocator import StrategyAllocator


def test_allocate_normalizes_confidence():
    alloc = StrategyAllocator()
    high = SimpleNamespace(symbol="AAPL", side="buy", confidence=2.5)
    low = SimpleNamespace(symbol="TSLA", side="sell", confidence=-0.5)

    # First call processes signals and normalizes confidence values
    alloc.allocate({"s": [high, low]})

    assert 0.5 <= high.confidence <= 1.0
    assert low.confidence == 0.01
