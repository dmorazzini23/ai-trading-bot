import types


def _resolve():
    # AI-AGENT-REF: prefer minimal allocator, fallback to script version
    try:
        import ai_trading.strategy_allocator as s
        return s
    except Exception:
        import scripts.strategy_allocator as s
        return s


def test_conf_gate_basic():
    mod = _resolve()
    Alloc = getattr(mod, "StrategyAllocator")
    cfg = types.SimpleNamespace(score_confidence_min=0.7)
    alloc = Alloc(config=cfg)

    lo = types.SimpleNamespace(symbol="A", side="buy", action="buy", confidence=0.55)
    hi = types.SimpleNamespace(symbol="B", side="buy", action="buy", confidence=0.85)

    out = alloc.allocate({"s": [lo, hi]})
    assert any(getattr(s, "symbol", "") == "B" for s in out)
    assert not any(getattr(s, "symbol", "") == "A" for s in out)

