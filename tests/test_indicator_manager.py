from ai_trading.indicator_manager import IndicatorManager, IndicatorSpec

def test_indicator_manager_streaming_basic():
    mgr = IndicatorManager()
    mgr.add("sma10", IndicatorSpec(kind="sma", period=10))
    mgr.add("ema10", IndicatorSpec(kind="ema", period=10))
    mgr.add("rsi14", IndicatorSpec(kind="rsi", period=14))
    out = None
    for i in range(1, 51):
        out = mgr.update(100 + i*0.1)
    assert out is not None
    assert set(out.keys()) == {"sma10", "ema10", "rsi14"}
    for v in out.values():
        assert isinstance(v, float)