from ai_trading.timeframe import TimeFrame, TimeFrameUnit, _safe_setattr


def test_timeframe_zero_arg_defaults():
    tf = TimeFrame()
    assert hasattr(tf, "amount")
    assert getattr(tf, "amount", None) == 1
    assert hasattr(tf, "unit")
    assert getattr(tf.unit, "name", "") == getattr(TimeFrameUnit.Day, "name", "Day")


def test_safe_setattr_handles_readonly_properties():
    class ReadOnly:
        @property
        def amount(self):
            return 7

    obj = ReadOnly()

    # Should not raise when attempting to set a read-only descriptor.
    _safe_setattr(obj, "amount", 42)

    assert obj.amount == 7
