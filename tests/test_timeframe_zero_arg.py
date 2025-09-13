from ai_trading.timeframe import TimeFrame, TimeFrameUnit


def test_timeframe_zero_arg_defaults():
    tf = TimeFrame()
    assert hasattr(tf, "amount")
    assert getattr(tf, "amount", None) == 1
    assert hasattr(tf, "unit")
    assert getattr(tf.unit, "name", "") == getattr(TimeFrameUnit.Day, "name", "Day")
