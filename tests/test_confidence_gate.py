import types

import ai_trading.strategy_allocator as sa


def test_conf_gate_basic():
    Alloc = getattr(sa, "StrategyAllocator")
    cfg = types.SimpleNamespace(score_confidence_min=0.7)
    alloc = Alloc(config=cfg)

    lo = types.SimpleNamespace(symbol="A", side="buy", action="buy", confidence=0.55)
    hi = types.SimpleNamespace(symbol="B", side="buy", action="buy", confidence=0.85)

    # First call builds history for confirmation
    alloc.allocate({"s": [lo, hi]})
    # Second call should confirm signals and apply confidence gating
    out = alloc.allocate({"s": [lo, hi]})
    assert any(getattr(s, "symbol", "") == "B" for s in out)
    assert not any(getattr(s, "symbol", "") == "A" for s in out)

