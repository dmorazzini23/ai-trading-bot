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


def test_allocate_blocks_sell_short_by_default(monkeypatch):
    monkeypatch.delenv("TRADING__ALLOW_SHORTS", raising=False)
    monkeypatch.delenv("AI_TRADING_ALLOW_SHORT", raising=False)
    alloc = StrategyAllocator()
    alloc.replace_config(delta_threshold=0.0, signal_confirmation_bars=1, min_confidence=0.0)
    signal = SimpleNamespace(symbol="MSFT", side="sell_short", confidence=0.8)

    out = alloc.allocate({"pairs": [signal]})

    assert out == []


def test_allocate_preserves_sell_short_side_when_canonical_short_policy_enabled(monkeypatch):
    monkeypatch.setenv("TRADING__ALLOW_SHORTS", "1")
    alloc = StrategyAllocator()
    alloc.replace_config(delta_threshold=0.0, signal_confirmation_bars=1, min_confidence=0.0)
    signal = SimpleNamespace(symbol="MSFT", side="sell_short", confidence=0.8)

    out = alloc.allocate({"pairs": [signal]})

    assert len(out) == 1
    assert out[0].side == "sell_short"


def test_allocate_blocks_sell_short_when_long_only_policy_set():
    alloc = StrategyAllocator()
    alloc.replace_config(
        delta_threshold=0.0,
        signal_confirmation_bars=1,
        min_confidence=0.0,
        allow_short=False,
    )
    signal = SimpleNamespace(symbol="MSFT", side="sell_short", confidence=0.8)

    out = alloc.allocate({"pairs": [signal]})

    assert out == []
