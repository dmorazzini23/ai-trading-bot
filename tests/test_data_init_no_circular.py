import sys


def test_import_timeutils_does_not_import_bars():
    sys.modules.pop("ai_trading.data.bars", None)
    import ai_trading.data.timeutils  # noqa: F401
    assert "ai_trading.data.bars" not in sys.modules
    from ai_trading.data import bars
    df = bars.empty_bars_dataframe()
    assert df.empty
