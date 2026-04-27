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


def test_allocate_preserves_sell_short_side():
    alloc = StrategyAllocator()
    alloc.replace_config(delta_threshold=0.0, signal_confirmation_bars=1, min_confidence=0.0)
    signal = SimpleNamespace(symbol="MSFT", side="sell_short", confidence=0.8)

    out = alloc.allocate({"pairs": [signal]})

    assert len(out) == 1
    assert out[0].side == "sell_short"
