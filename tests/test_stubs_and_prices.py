import pandas as pd


def test_timeframe_has_basic_members():
    from ai_trading.core.bot_engine import TimeFrame

    assert hasattr(TimeFrame, "Day")
    assert hasattr(TimeFrame, "Minute")
    assert hasattr(TimeFrame, "Hour")


def test_get_latest_close_handles_empty_and_variants():
    from ai_trading.utils.base import get_latest_close

    df_empty = pd.DataFrame(columns=["close"])
    assert get_latest_close(df_empty) == 0.0

    df_alt = pd.DataFrame({"Close": [None, "nan", 101.5]})
    assert get_latest_close(df_alt) == 101.5

    df_ok = pd.DataFrame({"close": [99.0, 100.0, float("nan")]})
    assert get_latest_close(df_ok) == 100.0
