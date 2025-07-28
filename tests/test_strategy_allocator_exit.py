from strategies import TradeSignal
import strategy_allocator


def test_exit_confirmation():
    alloc = strategy_allocator.StrategyAllocator()
    buy = TradeSignal(symbol="A", side="buy", confidence=1.0, strategy="s")
    sell = TradeSignal(symbol="A", side="sell", confidence=1.0, strategy="s")
    
    # Need to call allocate twice to confirm signals (signal_confirmation_bars = 2)
    alloc.allocate({"s": [buy]})  # First call to build history
    out1 = alloc.allocate({"s": [buy]})  # Second call should confirm and set hold_protect=3
    assert any(s.side == "buy" for s in out1)
    
    # Now try to sell - should be blocked by hold protection 3 times
    alloc.allocate({"s": [sell]})  # First sell call - builds history
    out2 = alloc.allocate({"s": [sell]})  # Second sell call - confirmed but blocked by hold_protect (remaining=2)
    assert not any(s.side == "sell" for s in out2)
    
    # Need to call sell 2 more times to exhaust hold protection
    out3 = alloc.allocate({"s": [sell]})  # Third sell call - blocked by hold_protect (remaining=1)  
    assert not any(s.side == "sell" for s in out3)
    
    out4 = alloc.allocate({"s": [sell]})  # Fourth sell call - blocked by hold_protect (remaining=0)
    assert not any(s.side == "sell" for s in out4)
    
    out5 = alloc.allocate({"s": [sell]})  # Fifth sell call - should finally go through
    assert any(s.side == "sell" for s in out5)
