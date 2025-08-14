def test_strategy_allocator_imports():
    import ai_trading.strategy_allocator as sa
    assert hasattr(sa, "StrategyAllocator")
