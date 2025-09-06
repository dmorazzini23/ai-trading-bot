from ai_trading.alpaca_api import get_timeframe_cls, get_timeframe_unit_cls


def test_timeframe_zero_arg_defaults():
    TimeFrame = get_timeframe_cls()
    unit_cls = get_timeframe_unit_cls()
    tf = TimeFrame()
    assert tf.amount == 1
    assert getattr(tf.unit, "name", "") == getattr(unit_cls.Day, "name", "Day")
